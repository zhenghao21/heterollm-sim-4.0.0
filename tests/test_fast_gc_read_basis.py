"""Live-GC read qualification uses physical tags, DAG order and multiplicity."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.fast_gc_read_basis import qualify_gc_read_basis
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
    assert result["initial_warm_read_pages_per_decode_upper"] == 512
    assert result["immutable_read_access_count"] > 56
    assert not bootstrap.kernel.physical_runtime.hbf_staging["hbf0.memory"]["controller"]._pages
