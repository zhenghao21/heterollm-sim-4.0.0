"""Native lowering does not price a cold device that live dispatch replaces."""
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]
from exploring_hbf_native_pilot import build_pilot_scenario
from heterollm_sim import data_motion, planner
from heterollm_sim.data_motion import endpoint_service, deferred_endpoint_construction
from heterollm_sim.planner import TopologyAwareBatchCostProvider
from heterollm_sim.reporting import run_scenario


def test_only_explicit_live_lowering_skips_cold_price():
    component = build_pilot_scenario("hbm_tp2").hardware.get_component("hbm0")
    with deferred_endpoint_construction():
        deferred = endpoint_service(component, 64, operation="read", name="test",
                                    page_offset_bytes=0, compact_preview=True)
        assert deferred.metadata["memory_accesses"][0]["byte_count"] == 64
        assert next(iter(deferred.metadata["physical_execution_by_owner"].values()))["physical_cost_deferred"]
        normal = endpoint_service(component, 64, operation="read", name="test", page_offset_bytes=0)
        assert not next(iter(normal.metadata["physical_execution_by_owner"].values())).get("physical_cost_deferred")
    normal = endpoint_service(component, 64, operation="read", name="test",
                              page_offset_bytes=0, compact_preview=True)
    assert not next(iter(normal.metadata["physical_execution_by_owner"].values())).get("physical_cost_deferred")


@pytest.mark.parametrize("target,method", [
    ("DramCore", "submit"), ("NandCore", "submit"), ("SramCore", "submit"),
    ("PhysicalService", "price"), ("PhysicalService", "_price")])
def test_unknown_construction_callbacks_keep_the_original_cold_execution(target, method, monkeypatch):
    from heterollm_sim.dram_core import DramCore
    from heterollm_sim.nand_core import NandCore
    from heterollm_sim.sram_core import SramCore
    owner = {"DramCore": DramCore, "NandCore": NandCore, "SramCore": SramCore,
             "PhysicalService": data_motion.PhysicalService}[target]
    original = getattr(owner, method)
    monkeypatch.setattr(owner, method, lambda *args, **kwargs: original(*args, **kwargs))
    component = build_pilot_scenario("hbm_tp2").hardware.get_component("hbm0")
    with deferred_endpoint_construction():
        assert data_motion._DEFER_ENDPOINT_CONSTRUCTION.get() is None
        result = endpoint_service(component, 64, operation="read", name="hook",
                                  page_offset_bytes=0, compact_preview=True)
    assert not next(iter(result.metadata["physical_execution_by_owner"].values())).get("physical_cost_deferred")


@pytest.mark.parametrize("operation,address,size", [("erase", 0, 64), ("read", -1, 64),
                                                    ("write", 10**20, 64), ("bad", 0, 64)])
def test_deferred_descriptor_still_rejects_invalid_access(operation, address, size):
    component = build_pilot_scenario("hbm_tp2").hardware.get_component("hbm0")
    with deferred_endpoint_construction(), pytest.raises(ValueError):
        endpoint_service(component, size, operation=operation, name="invalid",
                         page_offset_bytes=address, compact_preview=True)


def test_staged_hbf_retains_its_existing_construction_frontier(monkeypatch):
    from test_hbf_host_staging_integration import strict_staging_scenario
    scenario = strict_staging_scenario(all_hbf=True)
    provider = TopologyAwareBatchCostProvider(scenario, defer_physical_execution=True)
    assert provider._defer_endpoint_construction is False
    service = data_motion.resolve_service(scenario.hardware.get_component("hbf0"))
    original_frontier = {"service_ns": 50.25, "physical_cost_deferred": True}
    calls = []
    def price(*args, **kwargs):
        calls.append((args, kwargs))
        return original_frontier
    monkeypatch.setattr(data_motion.PhysicalService, "price", price)
    with deferred_endpoint_construction():
        result = data_motion._construction_service_bill(
            service, data_motion.AccessKind.READ, 64, page_offset_bytes=0)
    assert result is original_frontier
    assert len(calls) == 1


@pytest.mark.parametrize("case,identity", [("hbm_tp2", "hbm0"), ("hbf_weights_kv_tp1", "hbf0")])
def test_deferred_descriptor_uses_geometry_derived_capacity(case, identity):
    from heterollm_sim.physical_contract import require_physical_memory_config
    component = build_pilot_scenario(case).hardware.get_component(identity)
    config = replace(require_physical_memory_config(component), capacity_bytes=None)
    component = replace(component, capacity_bytes=config.effective_capacity_bytes,
        metadata={**component.metadata, "physical_memory_config": config})
    with deferred_endpoint_construction():
        result = endpoint_service(component, 64, operation="read", name="derived",
            page_offset_bytes=config.effective_capacity_bytes - 64, compact_preview=True)
        assert result.metadata["memory_accesses"][0]["byte_count"] == 64
        with pytest.raises(ValueError, match="capacity"):
            endpoint_service(component, 65, operation="read", name="derived",
                page_offset_bytes=config.effective_capacity_bytes - 64, compact_preview=True)


@pytest.mark.parametrize("kind", ["duplicate_owner", "foreign_bank", "foreign_command", "foreign_host"])
def test_authored_cross_owner_calendars_keep_cold_construction(kind, monkeypatch):
    from heterollm_sim.physical_contract import require_physical_memory_config
    scenario = build_pilot_scenario("hbm_tp2")
    component = scenario.hardware.get_component("hbm1" if kind != "foreign_host" else "hbf0")
    metadata = dict(component.metadata)
    if kind == "duplicate_owner":
        first = data_motion.resolve_service(scenario.hardware.get_component("hbm0")).physical_owner
        metadata["memory_service_owner"] = first
        metadata["memory_service"] = {**metadata.get("memory_service", {}), "physical_owner": first}
    else:
        field = {"foreign_bank": "bank_resource_prefix", "foreign_command": "shared_command_resource",
                 "foreign_host": "host_resource_id"}[kind]
        config = require_physical_memory_config(component)
        metadata["physical_memory_config"] = replace(config,
            metadata={**config.metadata, field: "another-owner:shared"})
    modified = replace(component, metadata=metadata)
    scenario = replace(scenario, hardware=replace(scenario.hardware, components=tuple(
        modified if row.component_id == component.component_id else row
        for row in scenario.hardware.components)))
    provider = TopologyAwareBatchCostProvider(scenario, defer_physical_execution=True)
    def observe(*args, **kwargs):
        assert data_motion._DEFER_ENDPOINT_CONSTRUCTION.get() is None
        return {"duration_ns": 0., "metadata": {}}
    monkeypatch.setattr(provider, "_estimate_exact_in_context", observe)
    provider.estimate(scenario, object())


def test_native_owner_prefixed_resources_and_one_shared_host_remain_eligible():
    from heterollm_sim.physical_contract import require_physical_memory_config
    scenario = build_pilot_scenario("hbm_tp2")
    components = []
    for component in scenario.hardware.components:
        if component.metadata.get("physical_memory_config") is not None:
            config = require_physical_memory_config(component)
            owner = data_motion.resolve_service(component).physical_owner
            if component.normalized_kind == "hbm":
                config = replace(config, metadata={**config.metadata,
                    "bank_resource_prefix": owner + ":private-bank",
                    "shared_command_resource": owner + ":private-command"})
            component = replace(component, metadata={**component.metadata, "physical_memory_config": config})
        components.append(component)
    scenario = replace(scenario, hardware=replace(scenario.hardware, components=tuple(components)))
    assert planner._private_physical_construction_context(scenario)
    context = planner.CompilationContext(scenario)
    with planner._compilation_scope(scenario, context):
        assert planner._private_physical_construction_context(scenario)
        assert planner._private_physical_construction_context(scenario)


@pytest.mark.parametrize("case", ["hbm_tp2", "hbf_kv_tp1", "hbf_weights_kv_tp1"])
def test_live_timing_and_physical_ledger_equal_priced_preview(case):
    scenario = build_pilot_scenario(case)
    scenario = replace(scenario, workload=replace(scenario.workload,
        requests=scenario.workload.requests[:1], request_count=1))
    priced = run_scenario(scenario, retention_policy="aggregate", batch_lowerer=
        TopologyAwareBatchCostProvider(scenario, defer_physical_execution=False))
    deferred = run_scenario(scenario, retention_policy="aggregate")
    assert priced.serving.events == deferred.serving.events
    assert priced.serving.request_metrics == deferred.serving.request_metrics
    assert len(priced.serving.batches) == len(deferred.serving.batches)
    for old, new in zip(priced.serving.batches, deferred.serving.batches):
        assert (old.start_ns, old.end_ns) == (new.start_ns, new.end_ns)
        for key in ("dram_traffic", "storage_traffic", "resource_busy_ns", "device_execution_ns"):
            assert old.cost.metadata.get(key) == new.cost.metadata.get(key)


def _ordered_resource_state(value):
    # Owner/resource identity, rather than unrelated simultaneous host
    # dispatch, defines the physical state. Cold preview removal may change
    # cross-owner ties and therefore a raw dictionary's insertion order.
    if type(value) is dict:
        return {key: _ordered_resource_state(value[key]) for key in sorted(value, key=repr)}
    if type(value) in (tuple, list):
        return type(value)(_ordered_resource_state(item) for item in value)
    return value


@pytest.mark.parametrize("tp,batch,architecture,hardware", [
    (2, 1, "dense", "HBF"), (8, 1, "dense", "HBF"),
    (8, 8, "dense", "HBF"), (2, 2, "maverick", "HBF+")])
def test_closed_native_construction_preserves_physical_results(tp, batch, architecture, hardware, monkeypatch):
    from exploring_hbf_native_config import build_native_comparison_scenario
    from heterollm_sim.serving import _OnlineRuntime
    from heterollm_sim.event_kernel import UnifiedEventKernel
    from test_parameterized_token_binding import assert_exact
    if architecture == "maverick":
        from test_native_maverick_semantics import miniature_maverick
        model = miniature_maverick(prompt=7).model
    else:
        model = build_pilot_scenario("hbm_tp1").model
    scenario = build_native_comparison_scenario(
        "llama3_405b", hardware, "SHORT", tp=tp, batch_size=batch,
        model_override=model,
        prompt_override=7 if architecture == "maverick" else 8,
        output_override=4, warmup_override=0,
        measurement_override=2,
    )
    scenario = replace(scenario, workload=replace(scenario.workload,
        requests=scenario.workload.requests[:batch], request_count=batch))
    assert planner._private_physical_construction_context(scenario)
    contexts, physical_records = [], []
    original_run = _OnlineRuntime.run
    def observe(online):
        contexts.append(online._execution_resource_kernel.physical_runtime)
        physical_records.append({})
        result = original_run(online)
        return result
    monkeypatch.setattr(_OnlineRuntime, "run", observe)
    original_resolve = UnifiedEventKernel._resolve_task_execution
    def observe_resolved(kernel, *args, **kwargs):
        result = original_resolve(kernel, *args, **kwargs)
        if contexts and kernel.physical_runtime is contexts[-1] and result[2]:
            metadata = result[0].metadata
            for owner, bill in metadata.get("physical_execution_by_owner", {}).items():
                physical_records[-1].setdefault(owner, []).append(deepcopy({
                    "accesses": tuple(row for row in metadata.get("memory_accesses", ())
                        if row.get("physical_owner") == owner),
                    "bill": bill,
                }))
        return result
    monkeypatch.setattr(UnifiedEventKernel, "_resolve_task_execution", observe_resolved)
    assert data_motion._native_construction_callbacks()
    provider = TopologyAwareBatchCostProvider(scenario, defer_physical_execution=True)
    provider._defer_endpoint_construction = False
    priced = run_scenario(scenario, retention_policy="aggregate", batch_lowerer=provider)
    deferred = run_scenario(scenario, retention_policy="aggregate")
    assert_exact(deferred.serving.events, priced.serving.events)
    assert_exact(deferred.serving.request_metrics, priced.serving.request_metrics)
    assert_exact([(batch.start_ns, batch.end_ns) for batch in deferred.serving.batches],
                 [(batch.start_ns, batch.end_ns) for batch in priced.serving.batches])
    for key in ("dram_traffic", "storage_traffic", "resource_busy_ns", "device_execution_ns",
                "category_time_ns", "critical_path_category_ns"):
        assert_exact([batch.cost.metadata.get(key) for batch in deferred.serving.batches],
                     [batch.cost.metadata.get(key) for batch in priced.serving.batches], key)
    snapshots = [context.snapshot() for context in contexts]
    for snapshot in snapshots:
        snapshot.pop("runtimes")
        snapshot.pop("allocators")
        for state in snapshot["runtime_state"].values():
            state.pop("active")
    assert physical_records[0] and physical_records[1]
    assert_exact(_ordered_resource_state(physical_records[1]),
                 _ordered_resource_state(physical_records[0]), "ordered_owner_transactions")
    assert_exact(_ordered_resource_state(snapshots[1]),
                 _ordered_resource_state(snapshots[0]), "physical_state_by_resource_id")
