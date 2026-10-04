from dataclasses import replace

from heterollm_sim.contracts import TaskCategory, TaskSpec, ResourceDemand
from heterollm_sim.data_motion import (
    AccessKind, DataAccess, MemoryPosition, expand_access,
    resolve_service,
)
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.memory_types import DramConfig


def _config():
    return DramConfig(channels=1, banks_per_group=1, rows_per_bank=4,
                      row_bytes=128, burst_bytes=64, open_ns=10,
                      read_latency_ns=5, burst_interval_ns=1,
                      lane_bandwidth_gb_s=64)


def _component():
    return ComponentSpec(component_id="dram0", kind="DDR", bandwidth_gbps=64,
                         metadata={"physical_memory_config": _config(),
                                   "physical_owner": "dram0",
                                   "memory_resource_id": "dram0"})


def test_expand_access_to_tasks_carries_formal_physical_descriptor():
    service = resolve_service(_component())
    expanded = expand_access(DataAccess("read", AccessKind.READ, 64,
                                        source=MemoryPosition("dram0", 0)),
                             {"dram0": service})
    task = expanded.to_tasks("request")[0]
    assert task.metadata["physical_memory_config"] == _config()
    assert task.metadata["memory_access"]["address"] == 0


def test_physical_task_preserves_unrelated_demand_and_delays_completion():
    config = _config()
    task = TaskSpec("read", "read", "read", TaskCategory.MEMORY,
                    demands=(ResourceDemand("gpu.compute", 1000),),
                    metadata={"physical_memory_config": config,
                              "memory_access": {"operation": "read", "address": 0,
                                                 "byte_count": 64,
                                                 "physical_owner": "dram0",
                                                 "resource_id": "dram0"}})
    kernel = UnifiedEventKernel.from_closed_graph((task,))
    event = kernel.step()
    assert event is not None
    assert event.end_ns >= 1000
    assert any(d.resource_id == "gpu.compute" for d in event.demands)
