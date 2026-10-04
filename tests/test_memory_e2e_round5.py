"""Round-five end-to-end contracts for physical memory.

The helpers below use the public endpoint and event-kernel adapters.  They are
deliberately small: the point is to catch calendar/counter/reporting seams,
not to duplicate the DRAM/NAND geometry tests.
"""

from dataclasses import replace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.data_motion import (
    AccessKind,
    PhysicalRuntimeContext,
    endpoint_service,
    resolve_physical_task,
    resolve_service,
)
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.ir import ComponentSpec, PortSpec
from heterollm_sim.memory_types import DramConfig, NandConfig
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.reporting import run_scenario


def _dram(**overrides):
    values = dict(
        channels=1, banks_per_group=1, rows_per_bank=8, row_bytes=128,
        burst_bytes=64, open_ns=10, close_ns=10, read_latency_ns=5,
        burst_interval_ns=1, lane_bandwidth_gb_s=64,
    )
    values.update(overrides)
    return DramConfig(**values)


def _nand(**overrides):
    values = dict(
        channels=1, targets_per_channel=1, dies_per_target=1, luns_per_die=1,
        planes_per_lun=1, blocks_per_plane=2, pages_per_block=4,
        page_bytes=1024, host_granularity_bytes=1024,
        host_bandwidth_gb_s=1024, internal_bandwidth_gb_s=1024,
        page_read_ns=10, page_program_ns=20, block_erase_ns=30,
    )
    values.update(overrides)
    return NandConfig(**values)


def _component(component_id, config, *, owner=None):
    owner = owner or component_id
    return ComponentSpec(
        component_id=component_id, kind="dram" if isinstance(config, DramConfig) else "ssd",
        ports=(PortSpec("mem", "PCIe", "device", version="5.0", lanes=1, bandwidth_gbps=512.0),),
        bandwidth_gbps=512.0, capacity_bytes=1024 * 1024,
        metadata={"physical_memory_config": config, "physical_owner": owner,
                  "memory_service": {"physical_bandwidth_gb_s": 64.0,
                                      "bandwidth_gb_s": 64.0}},
    )


def _task(task_id, config, *, owner="A", address=0, operation="read",
          preview=1.0, extra=(), accesses=None, dependencies=()):
    if accesses is None:
        accesses = {"operation": operation, "address": address,
                    "byte_count": 64 if isinstance(config, DramConfig) else 1024,
                    "physical_owner": owner, "resource_id": owner}
    return TaskSpec(
        task_id, task_id, task_id, TaskCategory.MEMORY,
        dependencies=tuple(dependencies),
        demands=(ResourceDemand(owner, preview, bytes_moved=64),) + tuple(extra),
        metadata={"physical_memory_config": config, "physical_owner": owner,
                  "memory_access": accesses},
    )


def _endpoint_task(task_id, component, *, address=0, read=True, owner=None):
    service = endpoint_service(component, 64, read=read, name=task_id,
                               page_offset_bytes=address)
    assert service is not None
    metadata = dict(service.metadata)
    if owner:
        metadata["physical_owner"] = owner
        metadata["memory_access"] = {**metadata["memory_access"], "physical_owner": owner}
    return TaskSpec(task_id, task_id, task_id, TaskCategory.MEMORY,
                    demands=service.demands, metadata=metadata)


def test_public_endpoint_to_kernel_warm_row_ignores_preview_and_keeps_gpu():
    component = _component("dram0", _dram())
    cold = _endpoint_task("cold", component)
    warm = replace(_endpoint_task("warm", component), dependencies=("cold",),
                   # This is the planner's logical placeholder.  It must be
                   # replaced by the endpoint's physical reservations rather
                   # than charged as an unrelated ordinary lane.
                       demands=(ResourceDemand("dram0", 10_000, bytes_moved=64),
                            ResourceDemand("gpu.compute", 1_000, work_units=1234)))
    kernel = UnifiedEventKernel.from_closed_graph((cold, warm))
    first, second = kernel.step(), None
    assert first is not None
    second = kernel.step()
    assert second is not None
    assert second.task.metadata["physical_completion_ns"] == pytest.approx(22)
    assert second.end_ns >= second.task.metadata["physical_completion_ns"]
    assert any(d.resource_id == "gpu.compute" for d in second.demands)
    # Changing only the stale planner preview cannot change the formal
    # physical completion produced by the endpoint descriptor.
    low_preview = replace(warm, demands=(ResourceDemand("dram0", 1, bytes_moved=64),
                                         ResourceDemand("gpu.compute", 1_000, work_units=1234)))
    retry = UnifiedEventKernel.from_closed_graph((cold, low_preview))
    retry.step()
    low = retry.step()
    assert low is not None
    assert low.task.metadata["physical_completion_ns"] == pytest.approx(22)
    assert low.end_ns == pytest.approx(second.end_ns)


@pytest.mark.parametrize("order", [("ordinary", "physical"), ("physical", "ordinary")])
def test_shared_ordinary_pcie_and_physical_host_use_one_calendar(order):
    config = _nand(metadata={"host_resource_id": "shared:pcie"})
    ordinary = TaskSpec(
        "ordinary", "ordinary", "ordinary", TaskCategory.COMMUNICATION,
        demands=(ResourceDemand("shared:pcie", 1000, bytes_moved=1000),),
    )
    physical = _task("physical", config, owner="ssd", preview=1,
                     accesses={"operation": "read", "address": 0,
                               "byte_count": 1024, "physical_owner": "ssd",
                               "resource_id": "ssd"})
    tasks = (ordinary, physical) if order[0] == "ordinary" else (physical, ordinary)
    kernel = UnifiedEventKernel.from_closed_graph(tasks)
    events = [kernel.step(), kernel.step()]
    assert all(events)
    by_id = {event.task.task_id: event for event in events}
    p = by_id["physical"]
    host = next(stage for stage in p.task.metadata["physical_resource_intervals"]
                if stage.name == "HOST_TRANSFER")
    o = by_id["ordinary"]
    if order[0] == "ordinary":
        assert host.start_ns >= o.end_ns
    else:
        # Physical NAND may perform array work before its host transfer.  The
        # ordinary PCIe interval may overlap that internal work, but never the
        # declared shared host-transfer reservation itself.
        assert o.end_ns <= host.start_ns or o.start_ns >= host.end_ns


def test_run_scenario_storage_probe_reports_real_mixed_counters_and_intervals():
    base = build_reference_scenario()
    components = list(base.hardware.components)
    index = next(i for i, item in enumerate(components) if item.component_id == "hostmem0")
    components[index] = replace(
        components[index],
        metadata={**components[index].metadata, "physical_memory_config": _dram(),
                  "memory_service": {"physical_bandwidth_gb_s": 64.0,
                                      "bandwidth_gb_s": 64.0}},
    )
    probe_ssd = _component("probe_ssd", _nand())
    scenario = replace(
        base,
        hardware=replace(base.hardware, components=tuple(components) + (probe_ssd,),
                          require_connected=False),
        workload=replace(
            base.workload, requests=base.workload.requests[:1], request_count=1,
            scheduler=replace(base.workload.scheduler, mode="static"),
            metadata={"storage_probe": (
                {"component_id": "hostmem0", "operation": "read", "byte_count": 64, "page_offset_bytes": 0},
                {"component_id": "hostmem0", "operation": "read", "byte_count": 64, "page_offset_bytes": 0},
                {"component_id": "hostmem0", "operation": "read", "byte_count": 64, "page_offset_bytes": 128},
                {"component_id": "probe_ssd", "operation": "read", "byte_count": 1024, "page_offset_bytes": 0},
                {"component_id": "probe_ssd", "operation": "write", "byte_count": 1024, "page_offset_bytes": 1024},
                {"component_id": "probe_ssd", "operation": "erase", "byte_count": 1024, "page_offset_bytes": 0},
            )},
        ),
    )
    result = run_scenario(scenario, retention_policy="exact")
    rows = [task for task in result.trace.tasks if task.metadata.get("storage_probe")]
    assert len(rows) == 6
    executions = [task.metadata["physical_execution"] for task in rows]
    assert executions[0]["row_misses"] == 1
    assert executions[1]["row_hits"] >= 1
    assert executions[2]["row_conflicts"] == 1
    assert executions[3]["pages_read"] >= 1
    assert executions[4]["pages_programmed"] >= 1
    assert executions[5]["erase_operations"] == 1
    assert sum(item["physical_read_bytes"] for item in executions) == 3 * 64 + 1024
    assert sum(item["physical_write_bytes"] for item in executions) == 1024
    assert any(interval.bytes_moved > 0 for row in rows for interval in row.resource_intervals)


def test_single_batch_and_mixed_public_results_have_same_counters():
    config = _dram()
    component = _component("dram0", config)
    service = resolve_service(component)
    single = service.price(AccessKind.READ, 64, page_offset_bytes=0,
                           runtime=PhysicalRuntimeContext(), arrival_ns=0)
    batch = service.price_batch(
        [{"request_id": "r", "operation": "read", "address": 0,
          "byte_count": 64, "arrival_ns": 0}], runtime=PhysicalRuntimeContext())
    assert batch["physical_read_bytes"] == single["physical_read_bytes"]
    assert batch["physical_bytes"] == single["physical_bytes"]
    mixed_accesses = (
        {"operation": "read", "address": 0, "byte_count": 64, "physical_owner": "A"},
        {"operation": "write", "address": 0, "byte_count": 64, "physical_owner": "A"},
    )
    mixed = resolve_physical_task(_task("mixed", config, accesses=mixed_accesses),
                                  PhysicalRuntimeContext(), 0)
    counters = mixed.metadata["physical_execution"]
    assert counters["logical_bytes"] == 128
    assert counters["physical_bytes"] == 128
    assert counters["physical_read_bytes"] == 64
    assert counters["physical_write_bytes"] == 64
    assert counters["operation_count"] == 2
    assert counters["burst_count"] == 2


def test_invalid_composite_is_atomic_and_kernel_retry_has_no_prefix_state():
    second = {"operation": "write", "address": -1, "byte_count": 64,
              "physical_owner": "A"}
    accesses = (
        {"operation": "read", "address": 0, "byte_count": 64, "physical_owner": "A"},
        second,
    )
    context = PhysicalRuntimeContext()
    with pytest.raises(ValueError, match="address"):
        resolve_physical_task(_task("composite", _dram(), accesses=accesses), context, 0)
    assert context.timeline.snapshot() == {}
    assert context.runtimes == {}

    task = _task("retry", _dram(), accesses=accesses)
    kernel = UnifiedEventKernel.from_closed_graph((task,))
    with pytest.raises(ValueError, match="address"):
        kernel.step()
    assert kernel.physical_runtime.timeline.snapshot() == {}
    second["address"] = 0
    event = kernel.step()
    assert event is not None
    assert event.task.metadata["physical_execution"]["operation_count"] == 2
