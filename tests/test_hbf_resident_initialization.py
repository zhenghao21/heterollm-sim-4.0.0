"""Warm HBF weights exist before measured dynamic media activity."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heterollm_sim import hbf_resident_init as initialization
from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.data_motion import PhysicalRuntimeContext
from test_hbf_staging_runtime import setup


def fixture(monkeypatch, sizes=(4096, 8192), *, alias=False):
    config, binding = setup()
    rows = [{"physical_owner": "flash", "buffer_id": "weights:" + str(index),
             "generation": 0, "size_bytes": size, "allocation_arena": "resident"}
            for index, size in enumerate(sizes)]
    preloads = [("flash", row["buffer_id"], 0) for row in rows]
    if alias:
        rows.append({**rows[0], "buffer_id": "weights:head", "alias_of": rows[0]["buffer_id"],
                     "alias_generation": 0, "alias_offset_bytes": 0})
        preloads.insert(0, ("flash", "weights:head", 0))
    tasks = (TaskSpec("weights", "bootstrap", "weight inventory", TaskCategory.MEMORY,
        metadata={"physical_owner": "flash", "physical_memory_config": config,
            "physical_memory_configs": {"flash": config},
            "hbf_host_staging_by_owner": {"flash": binding},
            "hbf_preloaded_buffers": tuple(preloads), "physical_allocations": tuple(rows)}),)
    monkeypatch.setattr(initialization, "_inventory_tasks", lambda scenario: tasks)
    component = NS(component_id="flash", normalized_kind="hbf", is_active_memory=True,
                   metadata={"memory_service_owner": "flash"})
    scenario = NS(weights_resident=True, hardware=NS(components=(component,)),
        model=NS(total_declared_weight_bytes=sum(sizes)),
        placement=NS(metadata={"control_plane": {"decision": {"rank_weight_shards": {
            "weights": [{"storage_component_id": "flash", "rank_id": index,
                         "replica_id": str(index), "physical_bytes": size}
                        for index, size in enumerate(sizes)]}}}}))
    kernel = NS(physical_runtime=PhysicalRuntimeContext())
    return scenario, kernel, tasks, config, binding


def test_preload_all_roots_and_padding_without_timed_io(monkeypatch):
    scenario, kernel, _, config, _ = fixture(monkeypatch)
    from heterollm_sim.memory_transfer import ResourceTimeline
    kernel.physical_runtime.timeline = ResourceTimeline()
    kernel.physical_runtime.timeline.reserve("bootstrap-sentinel", 100., 50.)
    calendar = kernel.physical_runtime.timeline.snapshot()
    initialization.initialize_hbf_resident_weights(scenario, kernel)
    state = kernel.physical_runtime.hbf_staging["flash"]
    controller = state["controller"]
    assert state["resident_initialization_complete"] is True
    assert state["resident_initialization_bytes"] == 12288
    assert state["preloaded"] == {("weights:0", 0), ("weights:1", 0)}
    assert controller.occupied_bytes == 2 * config.block_bytes
    assert controller.host_occupied_bytes == 0
    assert not controller._pages and not controller._used
    assert not any(key in controller.stats for key in ("pages_programmed", "flash_write_bytes", "gc_cycles"))
    assert all(frontier == config.pages_per_block for frontier in
               kernel.physical_runtime.runtimes["flash"].core._next_program_page.values())
    assert len(controller.programmed_page_counts) == 2
    assert kernel.physical_runtime.timeline.snapshot() == calendar
    assert not kernel.physical_runtime.runtimes["flash"].core._array_ready
    assert not kernel.physical_runtime.runtimes["flash"].core._buffer_ready


def test_alias_preloads_only_canonical_full_root(monkeypatch):
    scenario, kernel, _, config, binding = fixture(monkeypatch, alias=True)
    initialization.initialize_hbf_resident_weights(scenario, kernel)
    runtime = kernel.physical_runtime
    assert runtime.hbf_staging["flash"]["controller"].occupied_bytes == 2 * config.block_bytes
    assert runtime.allocators["flash"].canonical_range("weights:head", 0, 1) == ("weights:0", 0, 0)
    from heterollm_sim.hbf_staging_runtime import preload
    preload(runtime, "flash", "weights:head", 0, binding)
    assert len(runtime.hbf_staging["flash"]["preloaded"]) == 2


def test_unknown_late_weight_is_initialization_error(monkeypatch):
    scenario, kernel, _, _, binding = fixture(monkeypatch)
    initialization.initialize_hbf_resident_weights(scenario, kernel)
    runtime = kernel.physical_runtime
    runtime.allocators["flash"].allocate("late-head", 4096, allocation_arena="resident")
    from heterollm_sim.hbf_staging_runtime import preload
    with pytest.raises(ValueError, match="initialization inventory omitted"):
        preload(runtime, "flash", "late-head", 0, binding)
    assert ("late-head", 0) not in runtime.hbf_staging["flash"]["preloaded"]


def test_complete_weight_capacity_includes_all_block_padding_and_gc_reserve(monkeypatch):
    scenario, kernel, _, _, _ = fixture(monkeypatch, (8 * 16384, 8 * 16384))
    with pytest.raises(MemoryError, match="complete resident initialization and GC reserve"):
        initialization.initialize_hbf_resident_weights(scenario, kernel)
    assert not kernel.physical_runtime.hbf_staging["flash"]["preloaded"]
    assert not kernel.physical_runtime.hbf_staging["flash"]["controller"]._preloads


def test_partial_inventory_cannot_claim_initialization_complete(monkeypatch):
    scenario, kernel, _, _, _ = fixture(monkeypatch)
    scenario.placement.metadata["control_plane"]["decision"]["rank_weight_shards"]["weights"][0]["physical_bytes"] += 4096
    with pytest.raises(ValueError, match="inventory bytes disagree"):
        initialization.initialize_hbf_resident_weights(scenario, kernel)
    assert not kernel.physical_runtime.hbf_staging["flash"].get("resident_initialization_complete")


def test_preload_accepts_task_specific_routes_with_same_host_staging_identity(monkeypatch):
    scenario, kernel, tasks, _, binding = fixture(monkeypatch)
    other = {**binding, "routes": {**binding["routes"], "to_host": ()}}
    second = replace(tasks[0], task_id="different-initiator", metadata={**tasks[0].metadata,
        "hbf_host_staging_by_owner": {"flash": other}})
    monkeypatch.setattr(initialization, "_inventory_tasks", lambda _: (*tasks, second))
    initialization.initialize_hbf_resident_weights(scenario, kernel)
    assert kernel.physical_runtime.hbf_staging["flash"]["resident_initialization_complete"]


def test_missing_second_owner_is_rejected_even_when_global_bytes_match(monkeypatch):
    scenario, kernel, _, _, _ = fixture(monkeypatch)
    second = NS(component_id="second", normalized_kind="hbf", is_active_memory=True,
                metadata={"memory_service_owner": "flash2"})
    scenario.hardware.components += (second,)
    ledger = scenario.placement.metadata["control_plane"]["decision"]["rank_weight_shards"]["weights"]
    ledger[1]["storage_component_id"] = "second"
    with pytest.raises(ValueError, match="no HOST staging initialization binding: flash2"):
        initialization.initialize_hbf_resident_weights(scenario, kernel)
    assert not kernel.physical_runtime.hbf_staging


def test_wrong_owner_distribution_is_rejected_even_when_global_bytes_match(monkeypatch):
    scenario, kernel, tasks, config, binding = fixture(monkeypatch, (8192, 4096))
    second = NS(component_id="second", normalized_kind="hbf", is_active_memory=True,
                metadata={"memory_service_owner": "flash2"})
    scenario.hardware.components += (second,)
    ledger = scenario.placement.metadata["control_plane"]["decision"]["rank_weight_shards"]["weights"]
    ledger[0]["physical_bytes"], ledger[1]["physical_bytes"] = 4096, 8192
    ledger[1]["storage_component_id"] = "second"
    metadata = dict(tasks[0].metadata)
    metadata["hbf_host_staging_by_owner"] = {"flash": binding, "flash2": binding}
    metadata["physical_memory_configs"] = {"flash": config, "flash2": config}
    metadata["physical_allocations"] = (metadata["physical_allocations"][0],
        {**metadata["physical_allocations"][1], "physical_owner": "flash2"})
    metadata["hbf_preloaded_buffers"] = (("flash", "weights:0", 0), ("flash2", "weights:1", 0))
    monkeypatch.setattr(initialization, "_inventory_tasks", lambda _: (replace(tasks[0], metadata=metadata),))
    with pytest.raises(ValueError, match="inventory bytes disagree.*flash"):
        initialization.initialize_hbf_resident_weights(scenario, kernel)
    assert all(not state["preloaded"] for state in kernel.physical_runtime.hbf_staging.values())


@pytest.mark.parametrize("warm,kind", [(False, "hbf"), (True, "hbm")])
def test_cold_hbf_and_hbm_skip_inventory(monkeypatch, warm, kind):
    scenario, kernel, _, _, _ = fixture(monkeypatch)
    scenario.weights_resident = warm
    scenario.hardware.components[0].normalized_kind = kind
    monkeypatch.setattr(initialization, "_inventory_tasks", lambda _: pytest.fail("unexpected inventory lowering"))
    initialization.initialize_hbf_resident_weights(scenario, kernel)
    assert not kernel.physical_runtime.hbf_staging
    assert not kernel.physical_runtime.allocators


def test_dynamic_request_buffers_never_enter_preload_inventory(monkeypatch):
    scenario, kernel, tasks, _, _ = fixture(monkeypatch)
    row = {"physical_owner": "flash", "buffer_id": "kv:r:k", "generation": 0,
           "size_bytes": 131072, "allocation_arena": "resident"}
    metadata = {**tasks[0].metadata,
        "physical_allocations": (*tasks[0].metadata["physical_allocations"], row)}
    monkeypatch.setattr(initialization, "_inventory_tasks", lambda _: (replace(tasks[0], metadata=metadata),))
    initialization.initialize_hbf_resident_weights(scenario, kernel)
    assert kernel.physical_runtime.allocators["flash"].get_allocation("kv:r:k", 0) is None


def test_repeated_bootstrap_initialization_is_rejected(monkeypatch):
    scenario, kernel, _, _, _ = fixture(monkeypatch)
    initialization.initialize_hbf_resident_weights(scenario, kernel)
    with pytest.raises(ValueError, match="only once"):
        initialization.initialize_hbf_resident_weights(scenario, kernel)


@pytest.mark.parametrize("transactional", [False, True])
def test_snapshot_restore_preserves_complete_inventory_and_late_weight_guard(monkeypatch, transactional):
    from heterollm_sim import hbf_staging_runtime
    scenario, kernel, _, _, binding = fixture(monkeypatch)
    initialization.initialize_hbf_resident_weights(scenario, kernel)
    runtime = kernel.physical_runtime
    original = runtime.hbf_staging["flash"]
    occupied = original["controller"].occupied_bytes
    saved = hbf_staging_runtime.snapshot(runtime.hbf_staging, transactional=transactional)
    original["preloaded"].clear()
    original["resident_initialization_complete"] = False
    original["controller"].preload("temporary-test-weight", 0, 4096)
    runtime.hbf_staging = hbf_staging_runtime.restore(saved)
    restored = runtime.hbf_staging["flash"]
    assert restored["resident_initialization_complete"] is True
    assert restored["resident_initialization_scope"] == "ordinary_dense_analytic_weight_inventory"
    assert restored["resident_initialization_bytes"] == 12288
    assert restored["preloaded"] == {("weights:0", 0), ("weights:1", 0)}
    assert restored["controller"].occupied_bytes == occupied
    runtime.allocators["flash"].allocate("late-head", 4096, allocation_arena="resident")
    with pytest.raises(ValueError, match="initialization inventory omitted"):
        hbf_staging_runtime.preload(runtime, "flash", "late-head", 0, binding)


def test_custom_warm_hbf_lowerer_fails_before_bootstrap(monkeypatch):
    from heterollm_sim.reporting import run_scenario
    from test_hbf_host_staging_integration import strict_staging_scenario
    scenario = strict_staging_scenario(all_hbf=True)
    monkeypatch.setattr("heterollm_sim.reporting.bootstrap_control_plane",
                        lambda *args, **kwargs: pytest.fail("unexpected bootstrap"))
    with pytest.raises(ValueError, match="standard lowerer's complete resident"):
        run_scenario(scenario, batch_lowerer=lambda *args: None)


@pytest.mark.parametrize("model", ["qwen2_5-0_5b", "qwen2_5-1_5b", "qwen2_5-3b", "qwen2_5-7b"])
def test_formal_model_inventory_is_complete_without_request_execution(model):
    from b200_hbm_hbf_scan_config import build_scan_scenario
    from heterollm_sim.control_plane import plan_runtime_placement
    scenario = build_scan_scenario(model, "HBF", 262144, 1024)
    scenario = plan_runtime_placement(scenario).apply(scenario)
    kernel = NS(physical_runtime=PhysicalRuntimeContext())
    initialization.initialize_hbf_resident_weights(scenario, kernel)
    state = kernel.physical_runtime.hbf_staging["hbf0.memory"]
    assert state["resident_initialization_complete"]
    assert state["resident_initialization_bytes"] == scenario.model.total_declared_weight_bytes
    assert any("lm_head" in identity or "embedding" in identity for identity, _ in state["preloaded"])
    assert not state["controller"]._pages
    assert not state["controller"]._used


def test_hbf_without_placed_weights_or_accesses_needs_no_initialization(monkeypatch):
    scenario, kernel, _, _, _ = fixture(monkeypatch)
    scenario.placement.metadata["control_plane"]["decision"]["rank_weight_shards"] = {}
    monkeypatch.setattr(initialization, "_inventory_tasks", lambda _: ())
    initialization.initialize_hbf_resident_weights(scenario, kernel)
    assert not kernel.physical_runtime.hbf_staging


def test_shared_control_plane_bootstrap_initializes_before_request_execution():
    from b200_hbm_hbf_scan_config import build_scan_scenario
    from heterollm_sim.control_plane import bootstrap_control_plane
    scenario = build_scan_scenario("qwen2_5-0_5b", "HBF", 65536, 1024)
    bootstrap = bootstrap_control_plane(scenario, capture_physical_details=False)
    state = bootstrap.kernel.physical_runtime.hbf_staging["hbf0.memory"]
    assert bootstrap.result.completed_count > 0
    assert bootstrap.makespan_ns == bootstrap.result.makespan_ns
    assert state["resident_initialization_complete"]
    assert not state["controller"]._pages
    assert not state["controller"]._used
