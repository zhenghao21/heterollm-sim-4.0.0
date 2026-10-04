"""Round-four physical-memory checks through the public execution paths.

These tests intentionally exercise the production adapters and event kernel;
the direct-core tests in earlier rounds remain useful for geometry coverage,
but do not prove that a planner-produced task reaches the physical runtime.
"""

from dataclasses import replace
from collections.abc import Mapping

import pytest

from heterollm_sim.communication import TopologyRouter
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.data_motion import (
    AccessKind,
    DataAccess,
    MemoryPosition,
    LinkService,
    PhysicalRuntimeContext,
    expand_access,
    endpoint_service,
    resolve_service,
)
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.ir import ComponentSpec, HardwareSpec, LinkSpec, PortSpec
from heterollm_sim.memory_types import DramConfig, NandConfig
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.reporting import run_scenario


def _dram_config(**overrides):
    values = dict(
        channels=1,
        banks_per_group=1,
        rows_per_bank=4,
        row_bytes=128,
        burst_bytes=64,
        open_ns=10,
        close_ns=10,
        read_latency_ns=5,
        burst_interval_ns=1,
        lane_bandwidth_gb_s=64,
    )
    values.update(overrides)
    return DramConfig(**values)


def _nand_config(**overrides):
    values = dict(
        channels=1,
        targets_per_channel=1,
        dies_per_target=1,
        luns_per_die=1,
        planes_per_lun=1,
        blocks_per_plane=2,
        pages_per_block=4,
        page_bytes=1024,
        host_granularity_bytes=1024,
        host_bandwidth_gb_s=1024,
        internal_bandwidth_gb_s=1024,
        page_read_ns=10,
        page_program_ns=20,
        block_erase_ns=30,
    )
    values.update(overrides)
    return NandConfig(**values)


def _memory_component(component_id, kind, config, *, resource_id=None):
    resource_id = resource_id or f"{component_id}.memory"
    bandwidth = 1024.0 if kind.lower() in {"ssd", "hbf"} else 512.0
    return ComponentSpec(
        component_id=component_id,
        kind=kind,
        ports=(PortSpec(port_id="mem", protocol="PCIe", role="device", version="5.0", lanes=1, bandwidth_gbps=bandwidth),),
        bandwidth_gbps=bandwidth,
        capacity_bytes=1024 * 1024,
        metadata={
            "physical_memory_config": config,
            "memory_service": {
                "physical_bandwidth_gb_s": bandwidth / 8.0,
                "bandwidth_gb_s": bandwidth / 8.0,
                "resource_id": resource_id,
            },
        },
    )


def _physical_task(task_id, config, address, operation="read", owner="dram0"):
    return TaskSpec(
        task_id=task_id,
        request_id=task_id,
        name=task_id,
        category=TaskCategory.MEMORY,
        demands=(ResourceDemand(f"preview:{owner}", 1.0),),
        metadata={
            "physical_memory_config": config,
            "physical_owner": owner,
            "memory_access": {
                "operation": operation,
                "address": address,
                "byte_count": 64 if operation != "erase" else 1024,
                "physical_owner": owner,
            },
        },
    )


def test_real_router_expand_access_to_kernel_preserves_link_and_memory_demands():
    """A routed copy must retain endpoint, link and physical task metadata."""
    cfg = _dram_config()
    source = _memory_component("src", "dram", cfg)
    target = _memory_component("dst", "dram", cfg)
    hardware = HardwareSpec(
        name="round4",
        components=(source, target),
        links=(LinkSpec("src-dst", "src", "mem", "dst", "mem", "PCIe", bandwidth_gbps=512.0, latency_ns=3.0),),
        require_connected=True,
    )
    router = TopologyRouter(hardware)
    # The router path is independently checked before it is lowered into the
    # memory transaction API.  This catches resource loss at either boundary.
    phases = router.transfer_phases("src", "dst", 64, name="copy", source_page_offset_bytes=0, target_page_offset_bytes=0)
    assert phases and any(phase.metadata.get("link_id") == "src-dst" for phase in phases)

    services = {"src": resolve_service(source), "dst": resolve_service(target)}
    motion = expand_access(
        DataAccess("copy", AccessKind.COPY, 64, source=MemoryPosition("src", 0), target=MemoryPosition("dst", 0)),
        services,
        links={("src", "dst"): LinkService("src-dst", "link.src-dst", "link.src-dst", 64.0, 3.0)},
    )
    assert motion.phases
    assert any(d.resource_id for d in motion.demands)
    tasks = motion.to_tasks("request")
    assert tasks and any(task.metadata.get("memory_access") for task in tasks)
    kernel = UnifiedEventKernel.from_closed_graph(tasks, resource_owners=motion.resource_owners, resource_capacities=motion.resource_capacities)
    event = kernel.step()
    assert event is not None
    assert event.end_ns >= event.start_ns


def test_public_single_batch_and_kernel_have_equal_dram_completion():
    cfg = _dram_config()
    component = _memory_component("dram0", "dram", cfg)
    service = resolve_service(component)
    context = PhysicalRuntimeContext()
    single = service.price(AccessKind.READ, 64, page_offset_bytes=0, runtime=context, arrival_ns=0)
    batch_result = service.price_batch(
        [{"request_id": "r0", "operation": "read", "address": 0, "byte_count": 64, "arrival_ns": 0}],
        runtime=PhysicalRuntimeContext(),
    )
    # Implementations may return the queue summary alone or (summary, updated
    # runtime) when a caller supplies an explicit context.
    batch = batch_result[0] if isinstance(batch_result, tuple) and isinstance(batch_result[0], Mapping) else batch_result
    event = UnifiedEventKernel.from_closed_graph((_physical_task("r0", cfg, 0),)).step()
    assert event is not None
    assert single["completion_ns"] == pytest.approx(batch["requests"][0].completion_ns)
    assert event.end_ns == pytest.approx(single["completion_ns"])


def test_run_scenario_storage_probe_covers_row_repeat_conflict_and_nand_lifecycle():
    """The authored storage probes must use the canonical endpoint lowering."""
    base = build_reference_scenario()
    dram_cfg = _dram_config()
    nand_cfg = _nand_config()
    components = list(base.hardware.components)
    # hostmem0 remains connected to the reference graph, while its metadata is
    # replaced by a tiny DRAM transaction model for this reduced run.
    host_index = next(i for i, item in enumerate(components) if item.component_id == "hostmem0")
    host = components[host_index]
    components[host_index] = replace(
        host,
        metadata={
            **host.metadata,
            "physical_memory_config": dram_cfg,
            "memory_service": {"physical_bandwidth_gb_s": 64.0, "bandwidth_gb_s": 64.0},
        },
    )
    # A compact NAND endpoint is added to the same graph and probed directly;
    # no synthetic kernel-only task bypasses planner storage_probe lowering.
    nand = _memory_component("probe_ssd", "ssd", nand_cfg)
    hardware = replace(base.hardware, components=tuple(components) + (nand,), require_connected=False)
    probes = (
        {"component_id": "hostmem0", "operation": "read", "byte_count": 64, "page_offset_bytes": 0},
        {"component_id": "hostmem0", "operation": "read", "byte_count": 64, "page_offset_bytes": 0},
        {"component_id": "hostmem0", "operation": "read", "byte_count": 64, "page_offset_bytes": 128},
        {"component_id": "probe_ssd", "operation": "read", "byte_count": 1024, "page_offset_bytes": 0},
        {"component_id": "probe_ssd", "operation": "write", "byte_count": 1024, "page_offset_bytes": 1024},
        {"component_id": "probe_ssd", "operation": "erase", "byte_count": 1024, "page_offset_bytes": 0},
    )
    workload = replace(
        base.workload,
        requests=base.workload.requests[:1],
        request_count=1,
        scheduler=replace(base.workload.scheduler, mode="static"),
        metadata={"storage_probe": probes},
    )
    scenario = replace(base, hardware=hardware, workload=workload)
    result = run_scenario(scenario, retention_policy="exact")
    rows = [item for item in result.trace.tasks if item.metadata.get("storage_probe")]
    assert len(rows) == len(probes)
    executions = [item.metadata["physical_execution"] for item in rows]
    assert executions[0]["row_misses"] == 1
    assert executions[1]["row_hits"] >= 1
    assert executions[2]["row_conflicts"] == 1
    assert executions[3]["pages_read"] >= 1
    assert executions[4]["pages_programmed"] >= 1
    assert executions[5]["erase_operations"] == 1


def test_ordinary_endpoint_does_not_mark_physical_transaction():
    component = ComponentSpec(
        component_id="ordinary",
        kind="host_memory",
        ports=(PortSpec("mem", "DDR", "device", direction="bidirectional", version="5.0", lanes=1, bandwidth_gbps=64.0),),
        bandwidth_gbps=64.0,
        capacity_bytes=1024,
        metadata={"memory_service": {"physical_bandwidth_gb_s": 8.0, "bandwidth_gb_s": 8.0}},
    )
    service = endpoint_service(component, 64, read=True, name="ordinary")
    assert service is not None
    assert "memory_access" not in service.metadata
    assert not service.metadata.get("physical_memory_config")

