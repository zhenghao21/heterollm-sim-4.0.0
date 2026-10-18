"""Live-GC read qualification uses physical tags, DAG order and multiplicity."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.fast_gc_read_basis import (qualify_gc_read_basis,
    qualify_temporary_host_only_reads, temporary_read_media_bounds)
from heterollm_sim.hbf_staging import HbfStagingController
from heterollm_sim.memory_allocator import PhysicalAddressAllocator
from heterollm_sim.physical_presets import resolve_physical_config


def fixture(*, preload_pages=8):
    configured = resolve_physical_config("hbf_ocp_0_7", overrides={
        "channels": 1, "host_channels": 1, "planes_per_lun": 2,
        "blocks_per_plane": 4, "pages_per_block": 4,
        "capacity_bytes": 8 * 4 * 4096})
    controller = HbfStagingController(configured, 64 * 4096)
    allocator = PhysicalAddressAllocator(configured.capacity_bytes)
    allocator.allocate("weights", preload_pages * 4096)
    allocator.allocate("shared-head", preload_pages * 4096, alias_of="weights", alias_generation=0)
    controller.preload("weights", 0, preload_pages * 4096)
    physical = NS(hbf_staging={"flash": {"controller": controller}}, allocators={"flash": allocator})
    rows = tuple({"buffer_id": part, "bytes_per_token": 256} for part in ("kv:k", "kv:v"))
    def access(identity, size, *, offset=0, **updates):
        return {"operation": "read", "physical_owner": "flash", "buffer_id": identity,
                "allocation_generation": 0, "allocation_arena": "resident",
                "offset_bytes": offset, "byte_count": size, **updates}
    kv = TaskSpec("kv", "r", "kv", TaskCategory.MEMORY,
        metadata={"physical_memory_config": configured,
                  "memory_accesses": tuple(access(part, 4096) for part in ("kv:k", "kv:v"))})
    barrier = TaskSpec("barrier", "r", "arbitrary name", TaskCategory.MEMORY,
        dependencies=("kv",), metadata={"physical_memory_config": configured,
            "memory_accesses": (access("shared-head", preload_pages * 4096),)})
    return physical, controller, rows, (kv, barrier), access


def qualify(data, tasks=None, **updates):
    physical, _, rows, original, _ = data
    return qualify_gc_read_basis(tasks or original, physical, "flash", rows,
        context_tokens=16, token_boundary_completed=updates.pop("token_boundary_completed", True), **updates)


def test_real_preload_bank_tags_certify_barrier_independent_of_task_name():
    data = fixture()
    before = data[1].snapshot()
    result = qualify(data)
    assert result["qualified"], result
    assert result["kv_reads_per_page_per_decode_upper"] == 1
    assert result["barrier_distinct_tags_per_bank"] == (2, 2)
    assert result["initial_warm_read_pages_per_decode_upper"] == 4
    assert result["immutable_repeat_read_pages_per_decode_upper"] == 0
    assert data[1].snapshot() == before


def test_repeated_shared_canonical_reads_keep_access_and_page_multiplicity():
    data = fixture()
    _, _, _, tasks, access = data
    barrier = replace(tasks[1], metadata={**tasks[1].metadata, "memory_accesses": (
        access("weights", 4096), access("shared-head", 8 * 4096), access("weights", 4096))})
    result = qualify(data, (tasks[0], barrier))
    assert result["qualified"]
    assert result["immutable_read_access_count"] == 3
    assert result["immutable_read_pages_with_multiplicity"] == 10
    assert result["immutable_distinct_canonical_read_pages"] == 8
    assert result["immutable_repeat_read_pages_per_decode_upper"] == 2


def test_total_weight_size_cannot_replace_actual_per_bank_cache_tags():
    data = fixture(preload_pages=4)  # One physical block, all pages in bank 0.
    assert qualify(data)["reason"] == "gc_actual_immutable_bank_barrier_unavailable"


@pytest.mark.parametrize("change,reason", [
    ("no_dependency", "gc_actual_immutable_bank_barrier_unavailable"),
    ("kv_duplicate", "gc_kv_read_multiplicity_unqualified"),
    ("kv_alias", "gc_kv_read_is_not_unique_unaliased_full_prefix"),
    ("kv_partial", "gc_kv_read_is_not_unique_unaliased_full_prefix"),
    ("unknown_resident", "gc_immutable_read_canonical_identity_unavailable"),
    ("not_preloaded", "gc_immutable_read_has_no_actual_preload"),
])
def test_missing_cold_read_evidence_rejects(change, reason):
    data = fixture()
    physical, _, _, tasks, access = data
    kv, barrier = tasks
    if change == "no_dependency":
        barrier = replace(barrier, dependencies=())
    elif change == "kv_duplicate":
        kv = replace(kv, metadata={**kv.metadata, "memory_accesses": (
            *kv.metadata["memory_accesses"], kv.metadata["memory_accesses"][0])})
    elif change in {"kv_alias", "kv_partial"}:
        altered = dict(kv.metadata["memory_accesses"][0])
        altered.update({"alias_of": "other"} if change == "kv_alias" else {"offset_bytes": 64})
        kv = replace(kv, metadata={**kv.metadata,
            "memory_accesses": (altered, kv.metadata["memory_accesses"][1])})
    else:
        if change == "not_preloaded":
            physical.allocators["flash"].allocate("unknown", 4096)
        barrier = replace(barrier, metadata={**barrier.metadata,
            "memory_accesses": (access("unknown", 4096),)})
    assert qualify(data, (kv, barrier))["reason"] == reason


def test_unfinished_token_is_not_a_completed_cold_barrier():
    assert qualify(fixture(), token_boundary_completed=False)["reason"] == "gc_token_boundary_not_completed"


def test_mixed_gemm_barrier_is_valid_but_a_kv_alias_cannot_warm_hidden_tags():
    data = fixture()
    *_, tasks, access = data
    mixed = replace(tasks[1], metadata={**tasks[1].metadata, "memory_accesses": (
        access("temporary", 512, allocation_generation=3, allocation_arena="workspace"),
        *tasks[1].metadata["memory_accesses"],
        access("logits", 4096, operation="write", allocation_generation=3, allocation_arena="workspace"))})
    assert qualify(data, (tasks[0], mixed))["qualified"]
    aliased = replace(mixed, metadata={**mixed.metadata, "memory_accesses": (
        *mixed.metadata["memory_accesses"],
        access("hidden-kv-alias", 512, alias_of="kv:k", allocation_generation=3,
               allocation_arena="workspace"))})
    assert qualify(data, (tasks[0], aliased))["reason"] == "gc_unknown_persistent_or_aliased_read_state"


@pytest.mark.parametrize("phase", range(8))
def test_ordinary_prefix_budget_keeps_each_phase_and_matches_tail_endpoint(phase):
    from heterollm_sim.fast_state import _ordinary_program_prefix_bound
    current, remaining, fixed = 262144 + phase, 995, 274
    rows = [{"bytes_per_token": width} for width in (512, 512, 4096, 8192)]
    prefix = _ordinary_program_prefix_bound(rows, current, remaining, fixed)
    assert len(prefix) == remaining and list(prefix) == sorted(prefix)
    assert prefix[-1] == remaining * fixed + sum(
        ((current + remaining) * row["bytes_per_token"] // 4096
            - current * row["bytes_per_token"] // 4096) for row in rows)
    previous = 0
    for step, pages in enumerate(prefix, 1):
        assert pages - previous == fixed + sum(
            ((current + step) * row["bytes_per_token"] // 4096
                - (current + step - 1) * row["bytes_per_token"] // 4096) for row in rows)
        previous = pages


def test_formal_1_5b_bank_barrier_is_certified_from_preloads_without_request_execution():
    from b200_hbm_hbf_scan_config import build_scan_scenario
    from heterollm_sim.control_plane import bootstrap_control_plane
    from heterollm_sim.fast_state import _fixed_decode_write_geometry
    from heterollm_sim.planner import TopologyAwareBatchCostProvider, _execution_layers

    bootstrap = bootstrap_control_plane(build_scan_scenario("qwen2_5-1_5b", "HBF", 262144, 1024),
                                         capture_physical_details=False)
    scenario = bootstrap.scenario
    runner = NS(lowerer=TopologyAwareBatchCostProvider(scenario, defer_physical_execution=True),
        plan=NS(scenario=scenario), _execution_resource_kernel=bootstrap.kernel,
        _state_values=(NS(cached_tokens=262172, spec=scenario.workload.requests[0]),))
    rows = [{"buffer_id": f"kv:scan-request:{layer.layer_id}:rank0:tp0:pp0:{part}",
             "physical_owner": "hbf0.memory", "component_id": "hbf0", "bytes_per_token": 512}
            for layer in _execution_layers(scenario) for part in ("k", "v")]
    proof = _fixed_decode_write_geometry(runner, rows, 263167)
    result = proof["gc_read_qualification_by_owner"]["hbf0.memory"]
    assert result["qualified"], result
    assert result["kv_buffer_count"] == 56
    assert result["barrier_distinct_tags_per_bank"] == (2,) * 256
    assert result["initial_warm_read_pages_per_decode_upper"] == 0
    assert result["immutable_repeat_read_pages_per_decode_upper"] == 1
    warm = result["initial_warm_zero_proof"]
    assert warm["qualified"] and warm["fallback_pages_per_decode"] == 512
    assert warm["minimum_decode_context_tokens"] == 262145
    assert min(warm["distinct_non_tail_pages_before_tail_by_bank"]) == 254
    assert all(len(tags) == 2 for tags in warm["actual_tail_tags_by_bank"])
    assert result["immutable_read_access_count"] > 56
    media = proof["temporary_read_media_bounds_by_owner"]["hbf0.memory"]
    assert proof["temporary_read_page_upper_by_owner"]["hbf0.memory"] == 676
    assert media["rounded_read_pages_per_decode"] == 676
    assert media["potential_nand_read_pages_per_decode"] == 308
    assert media["proven_host_only_read_pages_per_decode"] == 368
    assert media["conservative_read_access_count"] == 0
    assert proof["temporary_host_only_dispatch_qualified"]
    assert not bootstrap.kernel.physical_runtime.hbf_staging["hbf0.memory"]["controller"]._pages


def temporary_span(offset=0, size=512, *, identity="temporary", generation=1, alias=None):
    return ("flash", identity, offset, size, generation, alias, None, 0)


def media(writes, reads):
    return temporary_read_media_bounds(writes, reads)["flash"]


def test_write_union_gap_outside_read_cannot_initialize_the_whole_page():
    result = media((temporary_span(), temporary_span(1024)), (temporary_span(),))
    assert result["rounded_read_pages_per_decode"] == 1
    assert result["potential_nand_read_pages_per_decode"] == 0
    assert result["proven_host_only_read_pages_per_decode"] == 1


def test_read_of_unknown_bytes_keeps_original_budget_even_on_incomplete_page():
    result = media((temporary_span(), temporary_span(1024)), (temporary_span(0, 1536),))
    assert result["potential_nand_read_pages_per_decode"] == 1
    assert result["proven_host_only_read_pages_per_decode"] == 0
    assert result["conservative_reasons"] == ("read_bytes_not_covered_by_known_write_union",)


def test_later_partial_write_completion_keeps_earlier_read_charged_order_independently():
    result = media((temporary_span(2048, 2048), temporary_span(0, 2048)),
                   (temporary_span(0, 512), temporary_span(4096, 64, identity="unknown")))
    assert result["rounded_read_pages_per_decode"] == 2
    assert result["potential_nand_read_pages_per_decode"] == 2
    assert result["proven_host_only_read_pages_per_decode"] == 0


def test_unaligned_span_can_leave_both_read_pages_host_only_and_retains_multiplicity():
    result = media((temporary_span(1, 4096),), (temporary_span(1, 4096),) * 3)
    assert result["rounded_read_pages_per_decode"] == 6
    assert result["potential_nand_read_pages_per_decode"] == 0
    assert result["proven_host_only_read_pages_per_decode"] == 6


@pytest.mark.parametrize("change", ["read_alias", "write_alias", "unknown_generation", "missing_write"])
def test_alias_or_unknown_fresh_source_retains_original_worst_alignment_budget(change):
    write = temporary_span(0, 4096)
    read = temporary_span(0, 4096)
    if change == "read_alias":
        read = temporary_span(0, 4096, alias="parent")
    elif change == "write_alias":
        write = temporary_span(0, 4096, alias="parent")
    elif change == "unknown_generation":
        read = temporary_span(0, 4096, generation=0)
    else:
        write = temporary_span(0, 4096, identity="other")
    result = media((write,), (read,))
    assert result["potential_nand_read_pages_per_decode"] == (2 if change == "read_alias" else 1)
    assert result["proven_host_only_read_pages_per_decode"] == 0
    assert result["conservative_read_access_count"] == 1


def test_any_aliased_write_prevents_claiming_another_base_buffer_is_host_only():
    result = media((temporary_span(), temporary_span(alias="temporary", identity="view")),
                   (temporary_span(),))
    assert result["potential_nand_read_pages_per_decode"] == 1


@pytest.mark.parametrize("condition", ["host_slots", "dispatch", "pressure", "rmw", "flush_hook"])
def test_host_only_reduction_never_crosses_partial_flush_guard(condition, monkeypatch):
    data = fixture()
    controller = data[1]
    bound = media((temporary_span(),), (temporary_span(),))
    assert qualify_temporary_host_only_reads(controller, bound,
        host_slots_sufficient=True, read_write_dispatch_qualified=True)["applied"]
    if condition in {"pressure", "rmw"}:
        controller.stats["pressure_flushes" if condition == "pressure" else "rmw_pages"] = 1
    elif condition == "flush_hook":
        monkeypatch.setattr(controller, "flush", lambda *args: 0.)
    proof = qualify_temporary_host_only_reads(controller, bound,
        host_slots_sufficient=condition != "host_slots",
        read_write_dispatch_qualified=condition != "dispatch")
    assert not proof["applied"]
    assert proof["charged_nand_read_pages_per_decode"] == 1


def test_real_partial_page_only_programs_under_pressure_or_explicit_flush():
    from test_hbf_staging import Hardware, config
    configured = config()
    bound = media((temporary_span(),), (temporary_span(),))
    # One data slot plus the permanently reserved GC slot.
    controller, hardware = HbfStagingController(configured, 2 * 4096), Hardware(configured, 2 * 4096)
    now = controller.write("temporary", 1, 0, 512, 0., hardware.io)
    now = controller.read("temporary", 1, 0, 512, now, hardware.io)
    assert not hardware.bytes("flash_write") and not hardware.bytes("flash_read")
    controller.write("another", 1, 0, 512, now, hardware.io)
    assert controller.stats["pressure_flushes"] == 1
    assert hardware.bytes("flash_write") == 4096
    proof = qualify_temporary_host_only_reads(controller, bound,
        host_slots_sufficient=True, read_write_dispatch_qualified=True)
    assert not proof["applied"]
    # Explicit flush has the same physical exception even with free slots;
    # such a dispatch cannot use the narrower static READ bound.
    controller, hardware = HbfStagingController(configured, 3 * 4096), Hardware(configured)
    now = controller.write("temporary", 1, 0, 512, 0., hardware.io)
    controller.flush("temporary", 1, now, hardware.io)
    assert hardware.bytes("flash_write") == 4096
    assert not qualify_temporary_host_only_reads(controller, bound,
        host_slots_sufficient=True, read_write_dispatch_qualified=False)["applied"]


def cold_fixture(*, preload_pages=8):
    from heterollm_sim.nand_core import NandCore
    physical, controller, rows, tasks, access = fixture(preload_pages=preload_pages)
    physical.runtimes = {"flash": NS(core=NandCore(controller.config))}
    def stable(task):
        return replace(task, metadata={**task.metadata, "memory_accesses": tuple(
            {**row, "address_source": "stable_buffer_tensor_offset"}
            for row in task.metadata.get("memory_accesses", ()))})
    embedding = TaskSpec("embedding", "r", "not a name based proof", TaskCategory.MEMORY,
        metadata={"memory_accesses": (access("weights", 4096,
            address_source="stable_buffer_tensor_offset"),)})
    kv = replace(tasks[0], dependencies=("embedding",))
    return physical, controller, rows, (embedding, kv, stable(tasks[1])), access


def cold_result(data, tasks=None):
    return qualify(data, tasks, minimum_decode_context_tokens=17)


def test_last_preload_tail_tags_are_cold_next_decode_and_repeat_is_not_removed():
    result = cold_result(cold_fixture())
    assert result["qualified"] and result["initial_warm_read_pages_per_decode_upper"] == 0
    assert result["immutable_repeat_read_pages_per_decode_upper"] == 1
    proof = result["initial_warm_zero_proof"]
    assert proof["qualified"]
    assert proof["minimum_decode_context_tokens"] == 17
    assert proof["fallback_pages_per_decode"] == 4
    assert proof["scope"] == "decode_cohort_with_completed_preceding_decode_cohort"
    assert proof["actual_tail_tags_by_bank"] == (((0, 2), (0, 3)), ((1, 2), (1, 3)))
    assert proof["distinct_non_tail_pages_before_tail_by_bank"] == (2, 2)


@pytest.mark.parametrize("change,reason", [
    ("non_ancestor", "initial_warm_other_immutable_read_not_dag_ancestor"),
    ("after_barrier_read", "initial_warm_final_scan_is_not_complete_canonical_preload"),
    ("early_tail_alias", "initial_warm_next_early_read_touches_previous_tail"),
    ("dynamic_offset", "initial_warm_dynamic_or_unknown_immutable_read_offset"),
    ("partial_head", "initial_warm_final_scan_is_not_complete_canonical_preload"),
    ("unknown_scope", "initial_warm_decode_scope_unavailable"),
    ("gc_preload", "initial_warm_immutable_preload_can_participate_in_gc"),
])
def test_missing_initial_warm_evidence_falls_back_without_relaxing_cold_kv_proof(change, reason):
    data = cold_fixture()
    physical, controller, _, original, access = data
    embedding, kv, barrier = original
    tasks = original
    if change == "non_ancestor":
        tasks = (embedding, replace(kv, dependencies=()), barrier)
    elif change == "after_barrier_read":
        later = replace(embedding, task_id="after", dependencies=(barrier.task_id,))
        tasks = (*original, later)
    elif change == "early_tail_alias":
        embedding = replace(embedding, metadata={"memory_accesses": (
            access("shared-head", 4096, offset=4 * 4096,
                   address_source="stable_buffer_tensor_offset"),)})
        tasks = (embedding, kv, barrier)
    elif change == "dynamic_offset":
        altered = {**embedding.metadata["memory_accesses"][0], "address_source": "dynamic_token_row"}
        tasks = (replace(embedding, metadata={"memory_accesses": (altered,)}), kv, barrier)
    elif change == "partial_head":
        altered = {**barrier.metadata["memory_accesses"][0], "byte_count": 7 * 4096}
        tasks = (embedding, kv, replace(barrier, metadata={"memory_accesses": (altered,)}))
    elif change == "gc_preload":
        controller._used[0] = 4
    result = (qualify(data, tasks) if change == "unknown_scope" else cold_result(data, tasks))
    assert result["qualified"], result
    assert result["initial_warm_read_pages_per_decode_upper"] == 4
    assert result["immutable_repeat_read_pages_per_decode_upper"] >= 1
    assert not result["initial_warm_zero_proof"]["qualified"]
    assert result["initial_warm_zero_proof"]["reason"] == reason


def test_every_bank_needs_k_distinct_non_tail_reads_before_any_old_tail():
    result = cold_result(cold_fixture(preload_pages=6))
    assert result["qualified"]
    assert result["initial_warm_read_pages_per_decode_upper"] == 4
    assert result["initial_warm_zero_proof"]["reason"] == "initial_warm_too_few_distinct_pages_before_tail"


@pytest.mark.parametrize("change", ["scalar_cache", "native_order", "staging_order"])
def test_custom_cache_or_page_submission_order_cannot_use_zero_warm(change, monkeypatch):
    from heterollm_sim import _compiled_nand, _io_execution
    data = cold_fixture()
    if change == "scalar_cache":
        monkeypatch.setattr(data[0].runtimes["flash"].core, "_cache_slot", lambda *args: None)
    elif change == "native_order":
        monkeypatch.setattr(_compiled_nand, "_run", lambda *args: None)
    else:
        monkeypatch.setattr(_io_execution, "_io_flash_pages", lambda *args: None)
    result = cold_result(data)
    assert result["qualified"] and result["initial_warm_read_pages_per_decode_upper"] == 4
    assert result["initial_warm_zero_proof"]["reason"] == "initial_warm_standard_ordered_lru_required"


def test_fragmented_uneven_preload_scan_tail_formula_matches_actual_page_order():
    from collections import deque
    from heterollm_sim.hbf_staging import _BlockPool
    data = cold_fixture(preload_pages=11)
    controller = data[1]
    controller.release("weights", 0)
    # The final partial group has one block; bank 0 also has a block in
    # the first group. Keep the real preload allocator and immutable map.
    controller._free_blocks = _BlockPool(0, 0)
    controller._free_blocks.extend_ranges((range(2, 3), range(5, 7), range(0, 2), range(3, 5)))
    controller.preload("weights", 0, 11 * 4096)
    result = cold_result(data)
    assert result["initial_warm_read_pages_per_decode_upper"] == 0, result
    proof = result["initial_warm_zero_proof"]
    descriptor = controller._preloads[("weights", 0)]
    tails, counts = [deque(maxlen=2) for _ in range(2)], [0, 0]
    for page_index in range(11):
        block, page = descriptor.page_location(page_index, 2, 4)
        tails[block % 2].append((block, page))
        counts[block % 2] += 1
    assert proof["actual_tail_tags_by_bank"] == tuple(tuple(tags) for tags in tails)
    assert proof["distinct_non_tail_pages_before_tail_by_bank"] == tuple(count - 2 for count in counts)


def test_initial_warm_proof_does_not_publish_or_change_python_native_state(monkeypatch):
    import pickle
    from heterollm_sim.nand_core import NandCore
    data = cold_fixture()
    controller, core = data[1], data[0].runtimes["flash"].core
    before = pickle.dumps((controller.snapshot(), vars(core)))
    def forbidden(*args):
        pytest.fail("a read-only cache proof must not publish native state")
    monkeypatch.setattr(NandCore, "_read_cache", property(forbidden))
    monkeypatch.setattr(NandCore, "_array_ready", property(forbidden))
    result = cold_result(data)
    assert result["initial_warm_zero_proof"]["qualified"]
    assert pickle.dumps((controller.snapshot(), vars(core))) == before
