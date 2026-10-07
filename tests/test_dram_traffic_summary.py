from types import SimpleNamespace

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.memory_types import DramConfig, MemoryKind
from heterollm_sim.planner import _summarize_dram_task_traffic
from heterollm_sim.reporting import _sum_batch_dram_traffic
from heterollm_sim.scalable_serving import execute_cost_schedule


def _config(**metadata):
    return DramConfig(
        kind=MemoryKind.GDDR, channels=1, banks_per_group=1,
        rows_per_bank=16, row_bytes=256, capacity_bytes=4096,
        interface_bandwidth_gb_s=32.0, metadata=metadata,
    ).__dict__


def test_compact_dram_summary_preserves_mixed_direction_and_owners():
    task = TaskSpec(
        task_id="mixed", request_id="report-test", name="output write",
        category=TaskCategory.MEMORY,
        demands=(ResourceDemand("gpu0.compute", 9.0, bytes_moved=1024),
                 ResourceDemand("gpu0.l2", 3.0, bytes_moved=512)),
        metadata={
            "physical_memory_config": _config(),
            "physical_memory_configs": {"memory-a.fabric": _config(), "memory-b.fabric": _config()},
            "physical_energy_pj_per_byte_by_owner": {"memory-a.fabric": 0.0, "memory-b.fabric": 0.0},
            "memory_accesses": (
                {"operation": "read", "address": 1, "byte_count": 70,
                 "physical_owner": "memory-a.fabric"},
                {"operation": "write", "address": 257, "byte_count": 3,
                 "physical_owner": "memory-b.fabric"},
            ),
        },
    )
    schedule = SimpleNamespace(tasks=(task,), resource_capacities={}, resource_owners={})
    full = execute_cost_schedule(schedule, retain_task_metadata=True)
    compact = execute_cost_schedule(schedule, retain_task_metadata=False)
    assert "memory_accesses" not in compact.execution_records[0].metadata
    assert compact.makespan_ns == full.makespan_ns
    assert compact.resource_busy_ns == full.resource_busy_ns
    full_ledger = _summarize_dram_task_traffic(full.execution_records)
    ledger = _summarize_dram_task_traffic(compact.execution_records)
    assert ledger == full_ledger
    assert ledger["logical_read_bytes"] == 70
    assert ledger["logical_write_bytes"] == 3
    assert ledger["logical_bytes"] == 73
    assert ledger["physical_read_bytes"] == 128
    assert ledger["physical_write_bytes"] == 64
    assert ledger["physical_bytes"] == 192
    assert {row["owner"] for row in ledger["resource_totals"].values()} == {
        "memory-a.fabric", "memory-b.fabric",
    }
    assert not {"gpu0.compute", "gpu0.l2"} & ledger["resource_totals"].keys()
    batch = SimpleNamespace(cost=SimpleNamespace(metadata={"dram_traffic": ledger}))
    report = _sum_batch_dram_traffic(SimpleNamespace(serving=SimpleNamespace(batches=(batch,))))
    assert report["logical_read_bytes"] == 70
    assert report["logical_write_bytes"] == 3
    assert report["owner_ids"] == ["memory-a.fabric", "memory-b.fabric"]


def test_dram_summary_uses_actual_resources_and_declared_owner():
    task = TaskSpec(
        task_id="read", request_id="report-test", name="write output",
        category=TaskCategory.MEMORY,
        metadata={
            "physical_memory_config": _config(
                bank_resource_prefix="opaque-bank", command_resource_prefix="opaque-command",
                data_resource_prefix="opaque-data",
            ),
            "memory_access": {"operation": "read", "address": 0, "byte_count": 1,
                              "physical_owner": "controller-a"},
        },
    )
    resolved = resolve_physical_task(task, PhysicalRuntimeContext(), 0.0)
    ledger = _summarize_dram_task_traffic(
        (resolved,), resource_owners={"opaque-data:0": "shared-controller"},
    )
    rows = ledger["resource_totals"]
    assert rows["opaque-data:0"]["owner"] == "shared-controller"
    assert rows["opaque-command:0"]["owner"] == "controller-a"
    assert ledger["logical_read_bytes"] == 1
    assert ledger["logical_write_bytes"] == 0
