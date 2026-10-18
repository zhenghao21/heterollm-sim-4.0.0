"""A GC forecast needs proved victim, read placement and complete IO bounds."""
from copy import deepcopy
from dataclasses import replace
from itertools import product

import pytest

from heterollm_sim.fast_gc import (
    _calendar_carry_in, _host_page_cost, _temporary_read_cache_accounting,
    _ordered_read_cache_contract, gc_service_bounds, project_gc_maintenance,
    qualify_gc_host_state, qualify_immutable_reads,
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
    return project_gc_maintenance(controller, **arguments)


def low_live_controller(live_per_block=1, *, controller=None, io=None):
    """Build the candidate pool through actual writes and lifetime releases."""
    controller = controller or HbfStagingController(config(), 64 * PAGE)
    io = io or callbacks()
    now = 0.
    pages = len(controller._free_blocks) * controller.config.pages_per_block + sum(
        controller.config.pages_per_block - controller._used[block] for block in controller._active)
    for page in range(pages):
        now = controller.write(("initial", page), 1, 0, PAGE, now, io)
    discarded = [key for owners in controller._owners.values()
                 for _, key in sorted(owners.items())[live_per_block:]]
    for buffer_id, generation, _ in discarded:
        controller.release(buffer_id, generation)
    assert not controller._active and not controller._free_blocks
    return controller, now


def live_services():
    result = services()
    result["gc_host_state_qualification"] = {"qualified": True, "physical_state_advanced": False}
    result["per_unit_costs"].update({
        "nand_read_page_serial_upper_ns": 2., "nand_program_page_serial_upper_ns": 3.,
        "host_read_page_serial_upper_ns": 5., "host_write_page_serial_upper_ns": 7.,
        "link_from_hbf_page_serial_upper_ns": 11., "link_to_hbf_page_serial_upper_ns": 13.,
        "read_cache_lru_slots_per_bank": 2,
        "kv_read_remap_page_serial_upper_ns": 17.,
        "persistent_cache_miss_page_serial_upper_ns": 2.,
        "gc_host_state_page_serial_upper_ns": 12.})
    return result


def cold_read_proof():
    return {"qualified": True, "reason": None, "physical_state_advanced": False,
        "kv_reads_per_page_per_decode_upper": 1,
        "immutable_repeat_read_pages_per_decode_upper": 3,
        "initial_warm_read_pages_per_decode_upper": 4}


def full_read_inventory():
    return {**cold_read_proof(),
        "method": "single_kv_prefix_read_and_actual_preload_bank_barrier/v1",
        "physical_io_executed": False,
        "kv_buffer_count": 2, "kv_reader_task_count": 1,
        "immutable_read_access_count": 3,
        "immutable_read_pages_with_multiplicity": 11,
        "immutable_distinct_canonical_read_pages": 8,
        "barrier_distinct_tags_per_bank": (2, 2),
        "barrier_all_kv_readers_are_dag_ancestors": True,
        "barrier_preloads_are_canonical": True,
        "barrier_mixed_temporary_accesses_cannot_warm_current_kv_tags": True,
        "token_boundary_completed": True}


def inventory_services():
    result = live_services()
    result["per_unit_costs"].update(ordinary_read_page_serial_upper_ns=9.,
                                   link_read_hbf_page_serial_upper_ns=3.)
    result["ordered_read_cache_contract"] = {"qualified": True,
        "physical_state_advanced": False,
        "same_bank_read_host_fifo": True, "read_host_resources_single_lane": True,
        "method": "standard_ordered_nand_lru_and_staging_read_dispatch/v1"}
    return result


def test_live_read_inventory_deduplicates_only_temporary_cache_surcharge():
    controller, _ = low_live_controller()
    bounds = inventory_services()
    before = deepcopy(bounds)
    tight = live_project(controller, gc_read_qualification=full_read_inventory(), service_bounds=bounds)
    legacy = live_project(controller, service_bounds=bounds)
    assert tight["qualified"] and tight["temporary_read_cache_accounting"]["applied"]
    assert not legacy["temporary_read_cache_accounting"]["applied"]
    assert tight["per_unit_costs"]["ordinary_read_page_serial_upper_ns"] == 5.
    assert legacy["per_unit_costs"]["ordinary_read_page_serial_upper_ns"] == 9.
    assert legacy["future_read_placement_serial_guard_ns"] - tight[
        "future_read_placement_serial_guard_ns"] == 5 * 3 * 4.
    for name in ("future_gc_read_remap_symmetric_guard_ns",
                 "future_gc_persistent_cache_symmetric_guard_ns",
                 "future_gc_host_state_symmetric_guard_ns", "future_gc_non_erase_serial_ns",
                 "future_gc_erase_serial_ns", "future_write_placement_serial_guard_ns"):
        assert tight[name] == legacy[name]
    assert bounds == before


@pytest.mark.parametrize("mutate", [
    lambda proof: proof.pop("method"),
    lambda proof: proof.update(physical_io_executed=True),
    lambda proof: proof.update(kv_reads_per_page_per_decode_upper=2),
    lambda proof: proof.update(immutable_repeat_read_pages_per_decode_upper=2),
    lambda proof: proof.update(immutable_distinct_canonical_read_pages=12),
    lambda proof: proof.update(barrier_all_kv_readers_are_dag_ancestors=False),
    lambda proof: proof.update(barrier_distinct_tags_per_bank=(2, 1)),
    lambda proof: proof.update(initial_warm_read_pages_per_decode_upper=1),
    lambda proof: proof.update(initial_warm_read_pages_per_decode_upper=0),
    lambda proof: proof.update(token_boundary_completed=False),
])
def test_incomplete_inventory_keeps_general_k_plus_one_bound(mutate):
    controller, _ = low_live_controller()
    proof, bounds = full_read_inventory(), inventory_services()
    mutate(proof)
    result = _temporary_read_cache_accounting(controller, proof,
        bounds["per_unit_costs"], bounds["ordered_read_cache_contract"])
    assert not result["applied"]
    assert result["legacy_ordinary_read_page_serial_upper_ns"] == 9.


@pytest.mark.parametrize("missing", [None, {}, {"qualified": False},
    {"qualified": True, "physical_state_advanced": True},
    {"qualified": True, "physical_state_advanced": False, "method": "custom"}])
def test_inventory_without_ordered_dispatch_keeps_general_bound(missing):
    controller, _ = low_live_controller()
    bounds = inventory_services()
    result = _temporary_read_cache_accounting(controller, full_read_inventory(),
        bounds["per_unit_costs"], missing)
    assert not result["applied"]
    assert result["reason"] == "standard_ordered_cache_dispatch_unqualified"


def test_inventory_bound_pays_lru_pollution_once_and_retains_repeated_hits():
    from itertools import permutations
    from test_fast_gc_budget import lru_misses
    # KV tags are unique and cold. Immutable repeated aliases are deliberately
    # retained in the hit inventory; no per-temporary k surcharge is needed
    # after paying every temporary miss and all possible persistent hits.
    persistent = ("kv0", "kv1", "weight0", "weight1", "weight0")
    repeated = 1
    for slots in (1, 2):
        for initial in permutations(("weight0", "weight1", "old_temporary"), slots):
            base = lru_misses(slots, initial, persistent)
            for first in range(len(persistent) + 1):
                for second in range(first, len(persistent) + 1):
                    actual = (persistent[:first] + ("temporary0",) + persistent[first:second]
                              + ("temporary1",) + persistent[second:])
                    assert lru_misses(slots, initial, actual) - base <= 2 + repeated + slots
    # With no repeat/initial-hit inventory, coefficient one alone is unsafe.
    assert lru_misses(2, ("weight0", "weight1"),
        ("temporary0", "weight0", "weight1")) == 3


def warm_zero_fixture(*, preload_pages=8):
    controller = HbfStagingController(config(), 64 * PAGE)
    controller.preload("weights", 0, preload_pages * PAGE)
    low_live_controller(controller=controller)
    proof = {"qualified": True, "physical_state_advanced": False,
        "minimum_decode_context_tokens": 65537, "fallback_pages_per_decode": 4,
        "scope": "decode_cohort_with_completed_preceding_decode_cohort",
        "final_full_canonical_preload_scan": True,
        "all_other_immutable_read_tasks_are_dag_ancestors": True,
        "standard_per_bank_lru_in_submission_order": True,
        "immutable_preload_excluded_from_gc": True,
        "bank_count": 2, "cache_slots_per_bank": 2,
        "actual_tail_tags_by_bank": (((0, 2), (0, 3)), ((1, 2), (1, 3))),
        "next_pre_scan_immutable_tail_overlap_pages": 0,
        "distinct_non_tail_pages_before_tail_by_bank": (2, 2)}
    return controller, proof


def live_project(controller, prefix=(0, 3, 6, 9, 12), **updates):
    parameters = dict(remaining_decode_steps=len(prefix), service_bounds=live_services(),
        ordinary_program_prefix_upper_by_decode=prefix, gc_read_qualification=cold_read_proof())
    parameters.update(updates)
    return project(controller, prefix[-1] if prefix else 0, **parameters)


def test_live_reservoir_selects_smallest_proved_q_and_records_every_candidate():
    controller, _ = low_live_controller()
    before = controller.snapshot()
    result = live_project(controller)
    assert result["qualified"] and result["live_projection_qualified"]
    assert result["live_copy_proven"] and not result["zero_copy_proven"]
    assert result["closed_victim_live_page_upper"] == 1
    assert result["closed_victim_live_histogram"] == (0, 7, 0, 0)
    assert result["closed_victim_reservoir_by_live_page_upper"] == (0, 7, 7, 7)
    assert result["gc_cycles_required_by_live_page_upper"] == (3, 4, 6, 12)
    assert result["cycles_upper"] == 4 and result["copied_pages_upper"] == 4
    assert result["future_gc_target_page_zero_erases_upper"] == 4
    assert controller.snapshot() == before


def test_q_two_and_headroom_adjust_each_cumulative_prefix_without_early_charging():
    controller, _ = low_live_controller(2)
    result = live_project(controller, (0, 2, 4, 6, 8))
    assert result["qualified"] and result["closed_victim_live_page_upper"] == 2
    assert result["gc_cycles_prefix_upper_by_decode"] == (0, 1, 2, 3, 4)
    assert result["future_gc_migrated_kv_page_token_upper"] == 20
    assert result["copied_pages_upper"] == 8


def test_live_complete_copy_chain_and_two_sided_read_costs_are_not_divided_by_banks():
    controller, _ = low_live_controller()
    result = live_project(controller)
    assert result["gc_cycles_prefix_upper_by_decode"] == (0, 1, 2, 3, 4)
    assert result["future_gc_migrated_kv_page_token_upper"] == 10
    assert result["future_gc_read_remap_symmetric_guard_ns"] == 10 * 17.
    assert result["future_gc_persistent_cache_symmetric_guard_ns"] == 5 * (3 + 4) * 2.
    assert result["future_gc_host_state_symmetric_guard_ns"] == 4 * (5. + 7.)
    assert result["future_gc_non_erase_serial_ns"] == 4 * 100.
    assert result["future_gc_erase_serial_ns"] == 4 * config().block_erase_ns
    assert result["gc_copy_nand_read_serial_upper_ns"] == 4 * 2.
    assert result["gc_copy_nand_write_serial_upper_ns"] == 4 * 3.
    assert result["gc_copy_host_read_serial_upper_ns"] == 4 * 5.
    assert result["gc_copy_host_write_serial_upper_ns"] == 4 * 7.
    assert result["gc_copy_link_from_hbf_serial_upper_ns"] == 4 * 11.
    assert result["gc_copy_link_to_hbf_serial_upper_ns"] == 4 * 13.
    assert result["gc_copy_cache_pollution_serial_upper_ns"] == 4 * 2 * 2.
    assert result["carry_in_serial_guard_ns"] == 7.
    assert result["gc_read_qualification"] == cold_read_proof()


def test_proved_zero_initial_warm_removes_only_initial_warm_cost_and_preserves_repeat_reads():
    controller, zero = warm_zero_fixture()
    positive = live_project(controller)
    proof = {**cold_read_proof(), "initial_warm_read_pages_per_decode_upper": 0,
             "initial_warm_zero_proof": zero}
    before = controller.snapshot()
    result = live_project(controller, gc_read_qualification=proof)
    assert result["qualified"] and result["live_projection_qualified"]
    assert result["immutable_repeat_read_pages_per_decode_upper"] == 3
    assert result["future_gc_persistent_cache_symmetric_guard_ns"] == 5 * 3 * 2.
    assert positive["future_gc_persistent_cache_symmetric_guard_ns"] - result[
        "future_gc_persistent_cache_symmetric_guard_ns"] == 5 * 4 * 2.
    for name in ("cycles_upper", "copied_pages_upper", "future_gc_erase_serial_ns",
            "future_gc_non_erase_serial_ns", "future_write_placement_serial_guard_ns",
            "future_read_placement_serial_guard_ns", "future_gc_read_remap_symmetric_guard_ns",
            "future_gc_host_state_symmetric_guard_ns"):
        assert result[name] == positive[name]
    assert controller.snapshot() == before


@pytest.mark.parametrize("zero", [None, {}, {"qualified": False},
    {"qualified": True, "physical_state_advanced": False}])
def test_zero_initial_warm_needs_the_complete_concrete_immutable_scan_proof(zero):
    controller, _ = warm_zero_fixture()
    proof = {**cold_read_proof(), "initial_warm_read_pages_per_decode_upper": 0,
             "initial_warm_zero_proof": zero}
    result = live_project(controller, gc_read_qualification=proof)
    assert not result["qualified"] and not result["live_projection_qualified"]
    assert result["reason"].startswith("live_gc_initial_warm_zero_proof_unavailable:")
    assert result["future_gc_persistent_cache_symmetric_guard_ns"] is None


@pytest.mark.parametrize("name,value", [
    ("physical_state_advanced", True), ("scope", "decode"),
    ("minimum_decode_context_tokens", 0), ("minimum_decode_context_tokens", True),
    ("minimum_decode_context_tokens", 65537.),
    ("fallback_pages_per_decode", 0), ("fallback_pages_per_decode", 4.),
    ("final_full_canonical_preload_scan", False),
    ("all_other_immutable_read_tasks_are_dag_ancestors", None),
    ("standard_per_bank_lru_in_submission_order", 1),
    ("immutable_preload_excluded_from_gc", False),
    ("bank_count", 1), ("bank_count", 2.), ("cache_slots_per_bank", 1),
    ("next_pre_scan_immutable_tail_overlap_pages", 1),
    ("next_pre_scan_immutable_tail_overlap_pages", False),
    ("distinct_non_tail_pages_before_tail_by_bank", (2,)),
    ("distinct_non_tail_pages_before_tail_by_bank", (2, 1)),
    ("distinct_non_tail_pages_before_tail_by_bank", (2, True)),
    ("actual_tail_tags_by_bank", (((0, 2), (0, 3)),)),
    ("actual_tail_tags_by_bank", (((0, 2),), ((1, 2), (1, 3)))),
    ("actual_tail_tags_by_bank", (((0, 2), (0, 2)), ((1, 2), (1, 3)))),
    ("actual_tail_tags_by_bank", (((1, 2), (0, 3)), ((1, 2), (1, 3)))),
    ("actual_tail_tags_by_bank", (((0, -1), (0, 3)), ((1, 2), (1, 3)))),
    ("actual_tail_tags_by_bank", (((0, 4), (0, 3)), ((1, 2), (1, 3)))),
    ("actual_tail_tags_by_bank", (((0, 2.), (0, 3)), ((1, 2), (1, 3)))),
    ("actual_tail_tags_by_bank", (((0, True), (0, 3)), ((1, 2), (1, 3)))),
    ("actual_tail_tags_by_bank", (((2, 2), (0, 3)), ((1, 2), (1, 3)))),
])
def test_zero_initial_warm_rejects_missing_scope_geometry_tags_and_order_evidence(name, value):
    controller, zero = warm_zero_fixture()
    proof = {**cold_read_proof(), "initial_warm_read_pages_per_decode_upper": 0,
        "initial_warm_zero_proof": {**zero, name: value}}
    result = live_project(controller, gc_read_qualification=proof)
    assert not result["qualified"] and not result["live_projection_qualified"]
    assert result["reason"].startswith("live_gc_initial_warm_zero_proof_unavailable:")
    assert result["future_gc_persistent_cache_symmetric_guard_ns"] is None


def test_zero_initial_warm_rejects_immutable_backing_that_can_enter_gc():
    controller, zero = warm_zero_fixture()
    controller._used[0] = 0
    result = live_project(controller, gc_read_qualification={**cold_read_proof(),
        "initial_warm_read_pages_per_decode_upper": 0, "initial_warm_zero_proof": zero})
    assert not result["qualified"]
    assert result["reason"].endswith("immutable_backing_can_participate_in_gc")


def test_zero_initial_warm_cannot_use_an_uninitialized_tail_of_a_partial_preload_block():
    controller, zero = warm_zero_fixture(preload_pages=6)
    result = live_project(controller, gc_read_qualification={**cold_read_proof(),
        "initial_warm_read_pages_per_decode_upper": 0, "initial_warm_zero_proof": zero})
    assert not result["qualified"]
    assert result["reason"].endswith("actual_tail_tag_not_initialized_by_preload")


def test_positive_initial_warm_fallback_ignores_unknown_zero_proof_and_keeps_full_cost():
    controller, _ = low_live_controller()
    result = live_project(controller, gc_read_qualification={**cold_read_proof(),
        "initial_warm_read_pages_per_decode_upper": 512, "initial_warm_zero_proof": None})
    assert result["qualified"]
    assert result["future_gc_persistent_cache_symmetric_guard_ns"] == 5 * (512 + 3) * 2.


@pytest.mark.parametrize("proof", [None, {}, {"qualified": False},
    {**cold_read_proof(), "physical_state_advanced": True}])
def test_live_without_independent_cold_read_proof_rejects_with_unknown_not_zero(proof):
    controller, _ = low_live_controller()
    result = live_project(controller, gc_read_qualification=proof)
    assert not result["qualified"] and not result["live_projection_qualified"]
    assert result["reason"] == "live_gc_cold_read_qualification_unavailable"
    assert result["future_gc_non_erase_serial_ns"] is None
    assert result["future_gc_read_remap_symmetric_guard_ns"] is None
    assert result["future_gc_persistent_cache_symmetric_guard_ns"] is None


@pytest.mark.parametrize("name,value", [
    ("kv_reads_per_page_per_decode_upper", 0),
    ("kv_reads_per_page_per_decode_upper", True),
    ("immutable_repeat_read_pages_per_decode_upper", -1),
    ("initial_warm_read_pages_per_decode_upper", None)])
def test_live_unknown_read_counts_cannot_receive_a_zero_remap_bound(name, value):
    controller, _ = low_live_controller()
    result = live_project(controller, gc_read_qualification={**cold_read_proof(), name: value})
    assert not result["qualified"] and result["reason"] == "live_gc_cold_read_counts_unqualified"
    assert result["future_gc_read_remap_symmetric_guard_ns"] is None


@pytest.mark.parametrize("prefix", [None, (0, 3, 6, 12), (0, 3, 6, 9, 13),
    (0, 6, 3, 9, 12), (0, 3, 6, -1, 12), (0, 3, 6, 9., 12)])
def test_live_cumulative_program_bound_must_cover_the_same_whole_tail(prefix):
    controller, _ = low_live_controller()
    result = live_project(controller, ordinary_program_prefix_upper_by_decode=prefix)
    assert not result["qualified"]
    assert result["reason"] == "live_gc_cumulative_program_bound_unqualified"
    assert result["future_gc_read_remap_symmetric_guard_ns"] is None


@pytest.mark.parametrize("name", ["gc_copy_page_serial_upper_ns",
    "nand_read_page_serial_upper_ns", "host_write_page_serial_upper_ns",
    "link_from_hbf_page_serial_upper_ns", "kv_read_remap_page_serial_upper_ns",
    "persistent_cache_miss_page_serial_upper_ns"])
def test_live_missing_service_component_is_unknown_not_a_zero_cost(name):
    controller, _ = low_live_controller()
    bounds = live_services()
    del bounds["per_unit_costs"][name]
    result = live_project(controller, service_bounds=bounds)
    assert not result["qualified"] and not result["live_projection_qualified"]
    assert result["reason"].startswith("live_gc_complete_service_bounds_unavailable:")
    assert result["future_gc_read_remap_symmetric_guard_ns"] is None


@pytest.mark.parametrize("proof", [None, {}, {"qualified": False},
    {"qualified": True, "physical_state_advanced": True}])
def test_live_without_host_state_proof_rejects_and_does_not_treat_it_as_zero(proof):
    controller, _ = low_live_controller()
    bounds = live_services()
    bounds["gc_host_state_qualification"] = proof
    result = live_project(controller, service_bounds=bounds)
    assert not result["qualified"] and not result["live_projection_qualified"]
    assert result["reason"] == "live_gc_host_state_qualification_unavailable"
    assert result["future_gc_host_state_symmetric_guard_ns"] is None


def test_live_without_sufficient_low_live_pool_cannot_use_a_global_diagnostic_q():
    controller, _ = low_live_controller(2)
    result = live_project(controller, (0, 8, 16, 24, 32))
    assert not result["qualified"] and result["copied_pages_upper"] is None
    assert result["cycles_upper"] is None
    assert result["closed_victim_live_histogram"] == (0, 0, 7, 0)
    assert result["reason"] == "live_gc_copy_remap_bound_unavailable"
    assert not result["diagnostic_live_copy_bound_used_for_acceptance"]


def test_empty_domain_keeps_original_costs_without_new_live_proof_or_prefix():
    result = live_project(exhausted(), gc_read_qualification=None,
                          ordinary_program_prefix_upper_by_decode=None)
    assert result["qualified"] and result["zero_copy_proven"]
    assert not result["live_copy_proven"] and not result["live_projection_qualified"]
    assert result["future_gc_read_remap_symmetric_guard_ns"] == 0.
    assert result["future_gc_persistent_cache_symmetric_guard_ns"] == 0.
    assert result["future_gc_host_state_symmetric_guard_ns"] == 0.
    assert result["future_gc_non_erase_serial_ns"] == 0.


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
    assert result["zero_copy_cycles_required"] == 8 and result["empty_victim_reservoir"] == 7
    assert result["reason"] == "live_gc_copy_remap_bound_unavailable"
    assert result["copied_pages_upper"] is None
    assert not result["diagnostic_live_copy_bound_used_for_acceptance"]


def test_empty_active_block_is_not_a_future_empty_victim_reservoir():
    controller = exhausted(pages=25)
    assert len(controller._active) == 1
    assert project(controller, 27, headroom=3)["qualified"]
    rejected = project(controller, 28, headroom=3)
    assert rejected["empty_victim_reservoir"] == 6
    assert rejected["zero_copy_cycles_required"] == 7 and not rejected["qualified"]


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


def real_physical(*, capture=False, host_updates=None):
    from heterollm_sim.data_motion import PhysicalRuntimeContext
    from heterollm_sim.hbf_staging_runtime import configure
    from test_hbf_staging_runtime import setup
    configured, binding = setup(capacity=16 * PAGE)
    if host_updates:
        binding = {**binding, "host_config": replace(binding["host_config"], **host_updates)}
    physical = PhysicalRuntimeContext(capture_details=capture)
    state = configure(physical, "flash", configured, binding)
    return physical, state["controller"], binding


@pytest.mark.parametrize("change", ["submit", "submit_read_pages", "_execute_accepted",
                                   "_execute_detailed_accepted", "_host_id",
                                   "instance_execute", "ordered_native_run",
                                   "_resident_scalar_eligible", "_submit_resident_scalar",
                                   "_resident_resource_contract", "_write_buffer_payload_bytes",
                                   "instance_resident_dispatch", "staging_page_journals"])
def test_temporary_dedup_requires_actual_standard_ordered_dispatch(monkeypatch, change):
    from heterollm_sim.nand_core import NandCore
    from heterollm_sim import _compiled_nand
    from heterollm_sim.memory_allocator import PhysicalAddressAllocator
    # Import/capture the canonical functions before introducing a hook.
    from heterollm_sim import fast_gc_read_basis
    physical, controller, _ = real_physical()
    physical.allocators.setdefault("flash", PhysicalAddressAllocator(controller.config.capacity_bytes))
    before = _ordered_read_cache_contract(physical, controller, "flash")
    assert before["qualified"], before
    replacement = lambda *args, **kwargs: pytest.fail("read-only proof executed a read")
    if change == "instance_execute":
        physical.runtimes["flash"].core._execute_accepted = replacement
    elif change == "instance_resident_dispatch":
        physical.runtimes["flash"].core._submit_resident_scalar = replacement
    elif change == "staging_page_journals":
        controller._full_write_page_journals = replacement
    elif change == "ordered_native_run":
        monkeypatch.setattr(_compiled_nand, "_run", replacement)
    else:
        monkeypatch.setattr(NandCore, change, replacement)
    rejected = _ordered_read_cache_contract(physical, controller, "flash")
    assert not rejected["qualified"]
    bounds = inventory_services()
    result = _temporary_read_cache_accounting(controller, full_read_inventory(),
        bounds["per_unit_costs"], rejected)
    assert not result["applied"]


@pytest.mark.parametrize("change", ["two_lanes", "outgoing_alias", "incoming_alias", "shared_slots"])
def test_temporary_dedup_refuses_non_fifo_or_hidden_read_host_lanes(change):
    from heterollm_sim.memory_allocator import PhysicalAddressAllocator
    physical, controller, _ = real_physical()
    physical.allocators.setdefault("flash", PhysicalAddressAllocator(controller.config.capacity_bytes))
    accepted = _ordered_read_cache_contract(physical, controller, "flash")
    assert accepted["qualified"], accepted
    name = accepted["read_host_resource_ids"][0]
    if change == "two_lanes":
        physical.timeline.lane_available[name] = [0., 0.]
    elif change == "outgoing_alias":
        physical.timeline.resource_aliases[name] = "another-client"
    elif change == "incoming_alias":
        physical.timeline.resource_aliases["another-client"] = name
    else:
        physical.timeline.lane_available[name] = physical.timeline.lane_available["another-client"] = [0.]
    refused = _ordered_read_cache_contract(physical, controller, "flash")
    assert not refused["qualified"]
    assert refused["reason"] == "nand_read_host_fifo_resources_or_lanes_unqualified"


@pytest.mark.parametrize("capture", [False, True])
def test_actual_nand_read_fifo_with_backlog_fits_temporary_and_hit_inventory(capture):
    from heterollm_sim.nand_core import NandCore
    configured = config()  # Two banks share one HOST channel, two LRU slots each.
    persistent = (0, 1, 2, 3, 0)
    def execute(trace):
        core = NandCore(configured, capture_details=capture)
        # Nonzero reader frontiers already occupy both cache slots per bank.
        for index in (24, 25, 26, 27):
            core.submit(AccessRequest("initial", Operation.READ, index * PAGE, PAGE, 100000.))
        end = 0.
        for index in trace:
            end = max(end, core.submit(AccessRequest("trace", Operation.READ,
                index * PAGE, PAGE, 0.)).completion_ns)
        return end
    baseline = execute(persistent)
    read_service = (configured.front_ns + configured.page_read_ns
        + configured.transfer_page_bytes / configured.internal_bandwidth_gb_s
        + PAGE * configured.host_channels / configured.host_bandwidth_for(Operation.READ))
    for first in range(len(persistent) + 1):
        for second in range(first, len(persistent) + 1):
            trace = persistent[:first] + (8,) + persistent[first:second] + (9,) + persistent[second:]
            # Two temporary accesses pay full service. The repeated persistent
            # page remains in the separate hit inventory, not discarded.
            assert execute(trace) - baseline <= 3 * read_service


@pytest.mark.parametrize("capture", [False, True])
def test_real_host_state_proof_records_actual_domain_and_preserves_every_live_object(capture):
    import pickle
    physical, controller, binding = real_physical(capture=capture)
    before = pickle.dumps(physical)
    proof = qualify_gc_host_state(physical, "host", (binding,))
    assert pickle.dumps(physical) == before
    assert proof["qualified"] and proof["physical_state_advanced"] is False
    assert proof["actual_config"]["control_mode"] == "analytical"
    assert proof["actual_config"]["scheduler"] == "fcfs"
    assert proof["actual_config"]["refresh_policy"] == "disabled"
    assert proof["actual_config"]["lane_count"] == 1
    bounds = gc_service_bounds(physical, controller, binding, 0.)
    units = bounds["per_unit_costs"]
    assert bounds["gc_host_state_qualification"] == proof
    assert units["gc_host_state_page_serial_upper_ns"] == (
        units["host_read_page_serial_upper_ns"] + units["host_write_page_serial_upper_ns"])


def test_formal_1_5b_host_state_service_is_qualified_after_bootstrap_without_request_execution():
    from pathlib import Path
    import sys
    from types import SimpleNamespace as NS
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from b200_hbm_hbf_scan_config import build_scan_scenario
    from heterollm_sim.control_plane import bootstrap_control_plane
    from heterollm_sim.fast_state import _fixed_decode_write_geometry
    from heterollm_sim.planner import TopologyAwareBatchCostProvider, _execution_layers
    bootstrap = bootstrap_control_plane(build_scan_scenario("qwen2_5-1_5b", "HBF", 262144, 1024),
                                        capture_physical_details=False)
    scenario, physical = bootstrap.scenario, bootstrap.kernel.physical_runtime
    runner = NS(lowerer=TopologyAwareBatchCostProvider(scenario, defer_physical_execution=True),
        plan=NS(scenario=scenario), _execution_resource_kernel=bootstrap.kernel,
        _state_values=(NS(cached_tokens=262172, spec=scenario.workload.requests[0]),))
    rows = [{"buffer_id": f"kv:scan-request:{layer.layer_id}:rank0:tp0:pp0:{part}",
             "physical_owner": "hbf0.memory", "component_id": "hbf0", "bytes_per_token": 512}
            for layer in _execution_layers(scenario) for part in ("k", "v")]
    geometry = _fixed_decode_write_geometry(runner, rows, 263167)
    state = physical.hbf_staging["hbf0.memory"]
    assert not state["controller"]._pages
    from heterollm_sim.fast_gc import _validate_initial_warm_zero_proof
    warm = geometry["gc_read_qualification_by_owner"]["hbf0.memory"]
    assert warm["initial_warm_read_pages_per_decode_upper"] == 0
    assert warm["immutable_repeat_read_pages_per_decode_upper"] == 1
    immutable_blocks = {block for preload in state["controller"]._preloads.values()
                        for span in preload.block_ranges for block in span}
    assert _validate_initial_warm_zero_proof(state["controller"],
        warm["initial_warm_zero_proof"], immutable_blocks) is None
    bounds = gc_service_bounds(physical, state["controller"],
        geometry["gc_staging_bindings_by_owner"]["hbf0.memory"], bootstrap.makespan_ns)
    proof = bounds["gc_host_state_qualification"]
    assert proof["qualified"], proof
    assert proof["actual_config"]["lane_count"] == 1
    units = bounds["per_unit_costs"]
    assert units["gc_host_state_page_serial_upper_ns"] == (
        units["host_read_page_serial_upper_ns"] + units["host_write_page_serial_upper_ns"])
    deduplicated = _temporary_read_cache_accounting(state["controller"], warm, units,
        bounds["ordered_read_cache_contract"])
    assert deduplicated["applied"], deduplicated
    assert deduplicated["ordinary_read_page_serial_upper_ns"] == (
        units["nand_read_page_serial_upper_ns"] + units["link_read_hbf_page_serial_upper_ns"])
    assert not state["controller"]._pages


@pytest.mark.parametrize("name", ["column_same_group_ns", "column_different_group_ns",
    "activate_same_group_ns", "activate_different_group_ns", "four_activate_window_ns",
    "write_to_read_same_group_ns", "write_to_read_different_group_ns",
    "read_to_write_ns", "write_to_read_ns"])
def test_live_host_nonzero_shared_history_rejects_but_zero_copy_budget_stays_usable(name):
    physical, controller, binding = real_physical(host_updates={name: 1.})
    bounds = gc_service_bounds(physical, controller, binding, 0.)
    proof = bounds["gc_host_state_qualification"]
    assert not proof["qualified"]
    assert proof["actual_config"][name] == 1.
    assert proof["reason"] == "gc_host_state_shared_timing_history_unqualified"
    assert bounds["per_unit_costs"]["gc_host_state_page_serial_upper_ns"] is None
    live, _ = low_live_controller()
    rejected = live_project(live, service_bounds=bounds)
    assert not rejected["qualified"]
    assert rejected["future_gc_host_state_symmetric_guard_ns"] is None
    empty = project(exhausted(), 4, service_bounds=bounds)
    assert empty["qualified"] and empty["future_gc_host_state_symmetric_guard_ns"] == 0.


@pytest.mark.parametrize("change,reason", [
    ("multiple_data_lanes", "gc_host_state_multiple_data_lanes_unqualified"),
    ("custom_namespace", "gc_host_state_custom_resource_namespace_unqualified"),
    ("outgoing_alias", "gc_host_state_resource_alias_unqualified"),
    ("incoming_alias", "gc_host_state_resource_alias_unqualified"),
    ("link_overlap", "gc_host_state_link_resource_overlap_unqualified"),
    ("multiple_resource_slots", "gc_host_state_resource_slots_unqualified"),
    ("shared_slot_list", "gc_host_state_resource_slots_unqualified"),
    ("bank_alias", "gc_host_state_bank_alias_unqualified"),
    ("custom_controller", "gc_host_state_custom_controller_unqualified")])
def test_host_shared_or_unknown_resources_cannot_receive_the_row_repair_bound(change, reason):
    updates = ({"data_lanes": 2} if change == "multiple_data_lanes" else
               {"metadata": {"data_resource_prefix": "shared-custom"}} if change == "custom_namespace" else None)
    physical, _, binding = real_physical(host_updates=updates)
    core, timeline = physical.runtimes["host"].core, physical.timeline
    data, command = "host:dram:data:0", "host:dram:command:0"
    if change == "outgoing_alias":
        timeline.resource_aliases = {data: "external"}
    elif change == "incoming_alias":
        timeline.resource_aliases = {"external": data}
    elif change == "link_overlap":
        binding = {**binding, "routes": {**binding["routes"], "to_hbf": ({
            **binding["routes"]["to_hbf"][0], "resource_id": data},)}}
    elif change == "multiple_resource_slots":
        timeline.lane_available[data] = [0., 0.]
    elif change == "shared_slot_list":
        timeline.lane_available[data] = timeline.lane_available[command] = [0.]
    elif change == "bank_alias":
        state = object()
        core.__dict__["_banks"] = {"a": state, "b": state}
    elif change == "custom_controller":
        core._data_id = lambda mapping: data
    proof = qualify_gc_host_state(physical, "host", (binding,))
    assert not proof["qualified"] and proof["reason"] == reason


@pytest.mark.parametrize("capture", [False, True])
def test_real_host_scratch_row_pollution_repair_fits_one_read_plus_write_page(capture):
    from heterollm_sim.dram_core import DramCore
    from heterollm_sim.memory_transfer import ResourceTimeline
    from heterollm_sim.memory_types import make_ddr_config
    configured = make_ddr_config(capacity_bytes=1024 * 1024)
    baseline = DramCore(configured, ResourceTimeline(), capture_details=capture)
    polluted = DramCore(configured, ResourceTimeline(), capture_details=capture)
    address = configured.bank_count * configured.row_bytes
    def access(core, operation, location, arrival):
        return core.submit(AccessRequest("host-state", operation, location, PAGE, arrival)).completion_ns
    baseline_start = access(baseline, Operation.READ, address, 0.)
    polluted_start = access(polluted, Operation.READ, address, 0.)
    scratch_end = access(polluted, Operation.WRITE, 0, polluted_start)
    scratch_end = access(polluted, Operation.READ, 0, scratch_end)
    baseline_end = access(baseline, Operation.READ, address, baseline_start)
    polluted_end = access(polluted, Operation.READ, address, scratch_end)
    repair = abs((polluted_end - scratch_end) - (baseline_end - baseline_start))
    assert repair > 0
    assert repair <= _host_page_cost(configured, Operation.READ) + _host_page_cost(configured, Operation.WRITE)


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
    projected = project_gc_maintenance(controller, program_pages=pages, free_program_pages=0,
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


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("retained_pages", [1, 2])
def test_live_reservoir_and_complete_service_bound_cover_real_gc(capture, retained_pages):
    from heterollm_sim._io_execution import _PhysicalIO
    physical, controller, binding = real_physical(capture=capture)
    io = _PhysicalIO(physical, "flash", physical.hbf_staging["flash"], binding, [])
    controller, cutoff = low_live_controller(retained_pages, controller=controller, io=io)
    core = physical.runtimes["flash"].core
    native = core.__dict__.get("_nand_numeric_state")
    arrays = None if native is None else tuple(array.copy() for array in native.arrays)
    flags = (core.__dict__["_nand_numeric_enabled"], core.__dict__["_nand_state_exposed"])
    controller_before = controller.snapshot()
    timeline_before = deepcopy(physical.timeline.__dict__)
    bound = gc_service_bounds(physical, controller, binding, cutoff)
    projected = live_project(controller, (3, 6, 9, 12), service_bounds=bound)
    assert projected["qualified"] and projected["live_projection_qualified"]
    assert projected["closed_victim_live_page_upper"] == retained_pages
    assert controller.snapshot() == controller_before
    assert physical.timeline.__dict__ == timeline_before
    assert (core.__dict__["_nand_numeric_enabled"], core.__dict__["_nand_state_exposed"]) == flags
    assert core.__dict__.get("_nand_numeric_state") is native
    if native is not None:
        import numpy as np
        for actual, expected in zip(native.arrays, arrays, strict=True):
            assert np.array_equal(actual, expected)
    gc_before = controller.stats.get("gc_cycles", 0)
    copies_before = controller.stats.get("gc_copied_pages", 0)
    start = now = cutoff + projected["carry_in_serial_guard_ns"]
    for step in range(4):
        now = controller.write(("future", step), 2, 0, 3 * PAGE, now, io)
        controller.release(("future", step), 2)
    actual_gc = controller.stats["gc_cycles"] - gc_before
    actual_copies = controller.stats["gc_copied_pages"] - copies_before
    assert 0 < actual_gc <= projected["cycles_upper"]
    assert 0 < actual_copies <= projected["copied_pages_upper"]
    assert now - start <= (projected["future_write_placement_serial_guard_ns"]
        + projected["future_gc_erase_serial_ns"] + projected["future_gc_non_erase_serial_ns"])
    # Full-copy unit includes all source NAND, scratch HOST and both LINK
    # directions; the persistent-cache miss component does not double LINK.
    costs = projected["per_unit_costs"]
    assert costs["persistent_cache_miss_page_serial_upper_ns"] == costs["nand_read_page_serial_upper_ns"]
    assert costs["kv_read_remap_page_serial_upper_ns"] == (
        costs["nand_read_page_serial_upper_ns"] + costs["link_read_hbf_page_serial_upper_ns"])


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
