"""Ordinary SRAM uses the native physical memory, allocation and event chain."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.data_motion import (AccessKind, PhysicalRuntimeContext, resolve_physical_task,
                                     resolve_service)
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.kv_pool import DynamicKVPool, KvPoolTransferRequest
from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
from heterollm_sim.memory_allocator import AllocationError
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest, MemoryKind, SramConfig, make_hbm_config
from heterollm_sim.physical_contract import (parse_execution_physical_config,
    physical_memory_bandwidth_caps_gbps, reconfigure_physical_config, require_physical_memory_config)
from heterollm_sim.sram_core import SramCore
from heterollm_sim.workspace_memory import physical_workspace_bound


def config(**overrides):
    values = dict(capacity_bytes=4096, burst_bytes=64, interface_bandwidth_gb_s=64.,
                  read_bandwidth_gb_s=64., write_bandwidth_gb_s=32.,
                  read_latency_ns=2., write_latency_ns=3., ports=2,
                  max_outstanding_requests=16)
    values.update(overrides)
    return SramConfig(**values)


def component(name="sram0", cfg=None):
    cfg = cfg or config()
    return ComponentSpec(name, "sram", capacity_bytes=cfg.capacity_bytes,
        read_bandwidth_gbps=cfg.read_bandwidth_gb_s * 8,
        write_bandwidth_gbps=cfg.write_bandwidth_gb_s * 8,
        bandwidth_gbps=cfg.interface_bandwidth_gb_s * 8,
        metadata={"physical_memory_config": cfg})


@pytest.mark.parametrize("field,value", [
    ("capacity_bytes", 0), ("capacity_bytes", 4097), ("burst_bytes", 0),
    ("ports", 0), ("ports", True), ("max_outstanding_requests", 0),
    ("interface_bandwidth_gb_s", float("inf")), ("read_bandwidth_gb_s", 65.),
    ("write_bandwidth_gb_s", 0.), ("read_latency_ns", -1.),
    ("write_latency_ns", float("nan")), ("kind", "HBM"), ("metadata", []),
])
def test_invalid_sram_contract_is_rejected(field, value):
    with pytest.raises((ValueError, TypeError)):
        config(**{field: value})


def test_sram_is_a_separate_kind_and_complete_serialized_contract():
    cfg = config()
    assert cfg.kind is MemoryKind.SRAM
    assert parse_execution_physical_config(asdict(cfg)) == cfg
    assert require_physical_memory_config(component(cfg=cfg)) == cfg
    assert physical_memory_bandwidth_caps_gbps(component(cfg=cfg)) == (512., 512., 256.)
    changed = reconfigure_physical_config(cfg, overrides={"capacity_bytes": 8192}, expand_geometry=True)
    assert changed.capacity_bytes == 8192
    assert changed.kind is MemoryKind.SRAM
    for missing in ("ports", "read_latency_ns", "interface_bandwidth_gb_s"):
        raw = asdict(cfg)
        del raw[missing]
        with pytest.raises(ValueError, match="missing explicit fields"):
            parse_execution_physical_config(raw)
    with pytest.raises(ValueError, match="physical_memory_config"):
        require_physical_memory_config(replace(component(), metadata={"physical_memory_config":
            make_hbm_config(capacity_bytes=cfg.capacity_bytes)}))


def test_directional_rates_shared_ports_and_pipelined_latency():
    cfg = config()
    read = SramCore(cfg).submit(AccessRequest("r", "read", 0, 128))
    write = SramCore(cfg).submit(AccessRequest("w", "write", 0, 128))
    assert read.completion_ns == 2. + 128 / 64.
    assert write.completion_ns == 3. + 128 / 32.
    assert read.physical_read_bytes == write.physical_write_bytes == 128
    assert read.pages_read == write.pages_programmed == write.erase_operations == 0
    core = SramCore(cfg)
    a = core.submit(AccessRequest("a", "read", 0, 64))
    b = core.submit(AccessRequest("b", "write", 64, 64))
    assert b.stages[-1].start_ns < a.completion_ns  # Different actual ports.
    c = core.submit(AccessRequest("c", "write", 0, 64))
    assert c.stages[-1].start_ns >= a.completion_ns  # Same read/write port.


def test_large_range_has_bounded_events_and_preserves_physical_byte_rounding():
    cfg = config(capacity_bytes=320_000_000, interface_bandwidth_gb_s=80000.,
                 read_bandwidth_gb_s=80000., write_bandwidth_gb_s=80000., ports=8,
                 read_latency_ns=1., write_latency_ns=1.)
    result = SramCore(cfg).submit(AccessRequest("large", "read", 0, cfg.capacity_bytes))
    assert result.completion_ns == 4001.
    assert result.transfer_bytes == cfg.capacity_bytes
    assert len(result.stages) == 9
    assert len(result.mapping) == 8
    partial = SramCore(cfg).submit(AccessRequest("partial", "write", 63, 2))
    assert partial.logical_bytes == 2
    assert partial.physical_write_bytes == 128
    assert partial.physical_read_bytes == 0  # SRAM has no NAND read-modify-write.


def test_queue_limit_and_invalid_request_do_not_mutate_live_state():
    core = SramCore(config(max_outstanding_requests=1))
    first = core.submit(AccessRequest("first", "read", 0, 128))
    second = core.submit(AccessRequest("second", "read", 0, 128))
    assert second.queue_wait_ns == first.completion_ns
    before = deepcopy(vars(core))
    with pytest.raises(ValueError, match="exceeds capacity"):
        core.submit(AccessRequest("outside", "read", core.config.capacity_bytes - 1, 2))
    assert vars(core) == before
    with pytest.raises(ValueError, match="ERASE"):
        core.submit(AccessRequest("erase", "erase", 0, 64))
    assert vars(core) == before


def test_native_runtime_snapshot_clone_and_allocator_reuse():
    cfg = config()
    memory = component(cfg=cfg)
    service = resolve_service(memory)
    runtime = PhysicalRuntimeContext(capture_details=False)
    active = runtime.runtime(cfg, service.physical_owner)
    assert isinstance(active.core, SramCore)
    allocator = runtime.allocators[service.physical_owner]
    block = allocator.allocate("scratch", cfg.capacity_bytes, 1)
    with pytest.raises(AllocationError):
        allocator.allocate("overflow", 64, 1)
    allocator.release("scratch", 1)
    assert allocator.allocate("reused", cfg.capacity_bytes, 2).base_address == block.base_address
    checkpoint = runtime.begin_transaction()
    bill = service.price(AccessKind.WRITE, 129, page_offset_bytes=0, runtime=runtime)
    assert bill["physical_write_bytes"] == 192
    assert bill["write_completion"] == "sram_port_write_complete"
    runtime.restore(checkpoint)
    again = service.price(AccessKind.WRITE, 129, page_offset_bytes=0, runtime=runtime)
    assert again == bill
    clone = deepcopy(runtime)
    clone_bill = service.price(AccessKind.READ, 128, page_offset_bytes=128, runtime=clone)
    live_bill = service.price(AccessKind.READ, 128, page_offset_bytes=128, runtime=runtime)
    assert live_bill == clone_bill
    assert live_bill["pages_touched"] == 0


def test_native_physical_task_reports_sram_and_workspace_lifetimes():
    cfg = config()
    task = TaskSpec("write", "request", "scratch-write", TaskCategory.MEMORY,
        metadata={"physical_owner": "scratch", "physical_memory_config": cfg,
                  "physical_allocations": ({"buffer_id": "activation", "size_bytes": 128,
                                             "generation": 1, "allocation_arena": "workspace"},),
                  "memory_accesses": ({"operation": "write", "buffer_id": "activation",
                      "byte_count": 128, "allocation_generation": 1},)})
    runtime = PhysicalRuntimeContext()
    bound = physical_workspace_bound((task,))
    assert bound["bytes_by_owner"] == {"scratch": 128}
    resolved = resolve_physical_task(task, runtime, 0.)
    row = resolved.metadata["physical_execution_by_owner"]["scratch"]
    assert row["kind"] == "SRAM"
    assert row["physical_write_bytes"] == 128
    assert row["resource_busy_ns"]
    assert isinstance(runtime.runtimes["scratch"].core, SramCore)
    from heterollm_sim.sram_traffic import summarize_sram_task_traffic
    traffic = summarize_sram_task_traffic((resolved,))
    assert traffic["physical_write_bytes"] == 128
    assert traffic["physical_read_bytes"] == 0
    assert traffic["owners"]["scratch"]["transaction_count"] == 1


def test_invalid_batch_restores_sram_ports_and_allocation_state():
    memory = component()
    service = resolve_service(memory)
    context = PhysicalRuntimeContext()
    context.runtime(config(), service.physical_owner)
    before = context.snapshot()
    with pytest.raises(ValueError, match="capacity"):
        service.price_batch(({"operation": "write", "address": 0, "byte_count": 128},
                             {"operation": "read", "address": 4090, "byte_count": 64}), runtime=context)
    assert context.snapshot() == before


def test_undeclared_lanes_and_nonfinite_timing_reject_before_sram_mutation():
    timeline = ResourceTimeline(lane_available={"sram:port:1": [0., 0.]})
    core = SramCore(config(), timeline)
    before = deepcopy(vars(core))
    with pytest.raises(ValueError, match="exactly one physical service lane"):
        core.submit(AccessRequest("bad-lanes", "read", 0, 128))
    assert vars(core) == before
    extreme = SramCore(config(read_latency_ns=1e308))
    before = deepcopy(vars(extreme))
    with pytest.raises(ValueError, match="finite timing range"):
        extreme.submit(AccessRequest("overflow", "read", 0, 64, 1e308))
    assert vars(extreme) == before


def test_custom_timeline_completion_is_not_replaced_by_the_sram_plan():
    class DelayedTimeline(ResourceTimeline):
        def transfer(self, resource_id, amount, earliest_ns, bandwidth_gb_s, **kwargs):
            return super().transfer(resource_id, amount, earliest_ns + 7., bandwidth_gb_s, **kwargs)
    core = SramCore(config(), DelayedTimeline())
    result = core.submit(AccessRequest("custom", "read", 0, 64))
    assert result.completion_ns == result.stages[-1].end_ns == 11.


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sram_kv_transfer_reuses_native_capacity_and_async_phase_contract(asynchronous):
    source, target = component("left"), component("right")
    executor = PhysicalKVTransferExecutor([source, target])
    page = SimpleNamespace(logical_page_id=0, request_id="request", layer_group="kv",
        owner_component="left", bytes=128, allocation_generation=0)
    executor.bind_page(page)
    request = KvPoolTransferRequest(0, "request", "kv", "left", "right", 128, 0, 0, 1)
    plan = executor.prepare_async(request) if asynchronous else executor(request)
    while plan.advance is not None:
        plan = plan.advance()
    plan.commit()
    snapshot = executor.snapshot()
    assert snapshot["bindings"][0].component_id == "right"
    assert executor.counters["read_bytes"] == executor.counters["write_bytes"] == 128
    assert all(isinstance(active.core, SramCore) for active in executor.runtime.runtimes.values())
    pool = DynamicKVPool([target], tokens_per_page=1, page_bytes=65)
    assert pool.allocation_bytes("right", 65) == 128


def test_short_native_gpu_request_uses_hbf_weights_and_ordinary_sram_without_hbm(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from exploring_hbf_compare import build_control_scenario
    from heterollm_sim.cost_models import HostMemoryProfile
    from heterollm_sim.reporting import run_scenario
    from heterollm_sim.serving import RequestStatus

    scenario = build_control_scenario("paper_hbf", prompt_tokens=16, output_tokens=3)
    cfg = config(capacity_bytes=320_000_000, interface_bandwidth_gb_s=80000.,
                 read_bandwidth_gb_s=80000., write_bandwidth_gb_s=80000., ports=8,
                 read_latency_ns=1., write_latency_ns=1., max_outstanding_requests=64,
                 metadata={"organization_basis": "eight globally pooled shared ports; not stack-local partitioning"})
    profile = HostMemoryProfile(bandwidth_gb_s=80000., read_bandwidth_gb_s=80000.,
        write_bandwidth_gb_s=80000., read_latency_ns=1., write_latency_ns=1.,
        transaction_bytes=64, max_outstanding_requests=64, parallel_lanes=8,
        resource_id="scratch.memory", name="ordinary-sram-test")
    previous = scenario.hardware.get_component("hbm0")
    memory = replace(component("scratch0", cfg), cost_profile_id="ordinary-sram-test",
        ports=tuple(replace(port, protocol="memory", version="1.0", bandwidth_gbps=640000.)
                    for port in previous.ports),
        metadata={"physical_memory_config": cfg, "memory_service_owner": "scratch.memory"})
    components = []
    for item in scenario.hardware.components:
        if item.component_id == "hbm0":
            components.append(memory)
        elif item.component_id == "gpu0":
            components.append(replace(item, ports=tuple(replace(port, protocol="memory", version="1.0",
                bandwidth_gbps=640000.) if port.port_id == "hbm0" else port for port in item.ports),
                metadata={**item.metadata, "scan_active_memory": "scratch0"}))
        else:
            components.append(item)
    links = tuple(replace(link,
        source_component="scratch0" if link.source_component == "hbm0" else link.source_component,
        target_component="scratch0" if link.target_component == "hbm0" else link.target_component,
        protocol="memory", version="1.0", bandwidth_gbps=0., latency_ns=0.,
        metadata={"service_ref": "scratch0.access"})
        if "hbm0" in (link.source_component, link.target_component) else link
        for link in scenario.hardware.links)
    registries = {kind: dict(values) for kind, values in scenario.component_profiles.items()}
    registries.setdefault("host_memory", {})["ordinary-sram-test"] = profile
    parallel = replace(scenario.placement.parallel, rank_mapping=tuple(
        replace(rank, memory_component_id="scratch0") for rank in scenario.placement.parallel.rank_mapping))
    scenario = replace(scenario,
        hardware=replace(scenario.hardware, components=tuple(components), links=links),
        component_profiles=registries, placement=replace(scenario.placement, parallel=parallel))
    live_contexts = []
    original = PhysicalRuntimeContext.runtime
    def observe(context, contract, owner):
        active = original(context, contract, owner)
        if isinstance(active.core, SramCore) and all(context is not value for value in live_contexts):
            live_contexts.append(context)
        return active
    monkeypatch.setattr(PhysicalRuntimeContext, "runtime", observe)
    result = run_scenario(scenario, retention_policy="aggregate")
    metric, = result.serving.request_metrics.values()
    assert metric.status is RequestStatus.FINISHED and metric.visible_output_tokens == 3
    assert not any(item.normalized_kind in {"hbm", "hbm_stack"}
                   for item in result.scenario.hardware.components)
    assert live_contexts
    assert any("sram:port" in resource and size > 0 for context in live_contexts
               for resource, size in context.timeline.bytes_moved.items())
    assert sum(batch.cost.metadata["sram_traffic"]["physical_write_bytes"]
               for batch in result.serving.batches) > 0
