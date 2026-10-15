"""Execution boundaries preserve one complete, stable physical configuration."""
from dataclasses import asdict, replace
from pathlib import Path
import sys

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
from heterollm_sim.memory_types import make_hbm_config


def physical_task(config):
    return TaskSpec("read", "request", "HBM read", TaskCategory.MEMORY,
        demands=(ResourceDemand("hbm0", 0., bytes_moved=64),),
        metadata={"physical_owner": "hbm0", "physical_memory_config": config,
            "memory_accesses": ({"operation": "read", "address": 0, "byte_count": 64},)})


@pytest.mark.parametrize("omitted", ["read_latency_ns", "channels"])
def test_direct_task_rejects_incomplete_mapping_without_default_execution(omitted):
    config = asdict(make_hbm_config(capacity_bytes=4096))
    del config[omitted]
    runtime = PhysicalRuntimeContext()
    before = runtime.snapshot()
    with pytest.raises(ValueError, match="missing explicit fields: " + omitted):
        resolve_physical_task(physical_task(config), runtime, 0.)
    assert runtime.snapshot() == before


def test_typed_and_complete_mapping_execute_identical_actual_dram_requests():
    config = make_hbm_config(capacity_bytes=4096, read_latency_ns=37.)
    outcomes = []
    for raw in (config, asdict(config)):
        runtime = PhysicalRuntimeContext()
        resolved = resolve_physical_task(physical_task(raw), runtime, 0.)
        physical = next(iter(resolved.metadata["physical_execution_by_owner"].values()))
        assert physical["physical_read_bytes"] == 64
        assert physical["physical_write_bytes"] == 0
        assert physical["completion_ns"] > 37.
        assert runtime.runtimes["hbm0"].core.config.read_latency_ns == 37.
        outcomes.append((resolved.demands, physical, runtime.timeline.bytes_moved))
    assert outcomes[0] == outcomes[1]


def test_metadata_order_and_live_config_roundtrip_reuse_same_core_and_allocator():
    config = make_hbm_config(capacity_bytes=4096)
    left = replace(config, metadata={"burst_mode": "independent", "note": "same"})
    right = replace(config, metadata={"note": "same", "burst_mode": "independent"})
    assert left == right
    runtime = PhysicalRuntimeContext()
    active = runtime.runtime(left, "hbm0")
    allocator = runtime.allocators["hbm0"]
    resolve_physical_task(physical_task(left), runtime, 0.)
    before = runtime.snapshot()
    assert runtime.runtime(right, "hbm0") is active
    assert runtime.runtime(active.core.config, "hbm0") is active
    assert runtime.runtime(asdict(active.core.config), "hbm0") is active
    assert runtime.allocators["hbm0"] is allocator
    assert runtime.snapshot() == before
    preview = runtime.preview_runtime(right, "hbm0")
    assert preview.core.config == active.core.config


def test_true_geometry_conflict_does_not_replace_existing_core_or_state():
    config = make_hbm_config(capacity_bytes=4096)
    runtime = PhysicalRuntimeContext()
    resolve_physical_task(physical_task(config), runtime, 0.)
    active = runtime.runtimes["hbm0"]
    before = runtime.snapshot()
    with pytest.raises(ValueError, match="multiple physical geometries"):
        runtime.runtime(replace(config, read_latency_ns=config.read_latency_ns + 1), "hbm0")
    assert runtime.runtimes["hbm0"] is active
    assert runtime.snapshot() == before


def test_authored_nested_metadata_cannot_mutate_live_core_contract():
    nested = {"annotations": {"source": ["original"]}}
    config = make_hbm_config(capacity_bytes=4096, metadata=nested)
    runtime = PhysicalRuntimeContext()
    active = runtime.runtime(config, "hbm0")
    config.metadata["annotations"]["source"].append("changed")
    assert active.core.config.metadata["annotations"] == {"source": ["original"]}
    with pytest.raises(ValueError, match="multiple physical geometries"):
        runtime.runtime(config, "hbm0")
    assert runtime.runtime(active.core.config, "hbm0") is active


@pytest.mark.parametrize("serialized", [False, True])
def test_kv_executor_uses_same_validated_contract_as_shared_runtime(serialized):
    config = make_hbm_config(capacity_bytes=4096, read_latency_ns=37.)
    raw = asdict(config) if serialized else config
    component = ComponentSpec("hbm0", "hbm", capacity_bytes=4096,
        metadata={"physical_memory_config": raw, "memory_service_owner": "hbm0"})
    runtime = PhysicalRuntimeContext()
    active = runtime.runtime(config, "hbm0")
    executor = PhysicalKVTransferExecutor([component], runtime=runtime)
    assert executor._configs["hbm0"] == config
    assert runtime.runtimes["hbm0"] is active
    incomplete = asdict(config)
    del incomplete["read_latency_ns"]
    with pytest.raises(ValueError, match="missing explicit fields: read_latency_ns"):
        PhysicalKVTransferExecutor([replace(component,
            metadata={**component.metadata, "physical_memory_config": incomplete})], runtime=runtime)
    assert runtime.runtimes["hbm0"] is active


def test_serving_revalidates_relay_contract_before_compilation():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from hbf_hbm_ablation_paths import build_path_scenario
    from heterollm_sim.serving import compile_serving_plan

    scenario, _ = build_path_scenario("relay", mixed_weights=False, prompt=8, output=2)
    compile_serving_plan(scenario)
    # Component metadata can be edited after ScenarioConfig validation. The
    # serving admission boundary must not treat mere config presence as valid.
    relay = scenario.hardware.get_component("relaymem")
    raw = dict(relay.metadata["physical_memory_config"])
    del raw["read_latency_ns"]
    relay.metadata["physical_memory_config"] = raw
    with pytest.raises(ValueError, match="missing explicit fields: read_latency_ns"):
        compile_serving_plan(scenario)


def test_pool_ir_capacity_contract_and_smaller_logical_budget_are_distinct():
    from heterollm_sim.kv_pool import DynamicKVPool, KvPoolComponent

    config = make_hbm_config(capacity_bytes=4096)
    physical = ComponentSpec("hbm0", "hbm", capacity_bytes=4096,
        metadata={"physical_memory_config": config})
    assert DynamicKVPool([physical], page_bytes=1)._allocation_granularity["hbm0"] == config.burst_bytes
    with pytest.raises(ValueError, match="capacity_bytes must be positive and equal"):
        DynamicKVPool([replace(physical, capacity_bytes=2048)])
    logical = KvPoolComponent("hbm0", 2048, kind="hbm",
        metadata={"physical_memory_config": asdict(config)})
    pool = DynamicKVPool([logical], page_bytes=1)
    assert pool.components["hbm0"].capacity_bytes == 2048
    assert pool._allocation_granularity["hbm0"] == config.burst_bytes
    incomplete = asdict(config)
    del incomplete["burst_bytes"]
    with pytest.raises(ValueError, match="missing explicit fields: burst_bytes"):
        DynamicKVPool([replace(logical, metadata={"physical_memory_config": incomplete})])
    assert DynamicKVPool([KvPoolComponent("logical", 2048)])._allocation_granularity["logical"] == 1
