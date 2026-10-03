from dataclasses import replace

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.planner import _summarize_nand_task_traffic
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.reporting import report_dict, run_scenario


def _ledger():
    task = TaskSpec(
        "nand.read",
        "request",
        "nand.read",
        TaskCategory.MEMORY,
        demands=(ResourceDemand("component.hbf.read", 120.0, bytes_moved=8192, energy_pj=3.0),),
        metadata={
            "phase_metadata": {
                "direct_memory_access": {
                    "physical_owner": "hbf.owner",
                    "read_bytes": 64,
                    "memory_service": {
                        "operation": "read",
                        "host_transfer_bytes": 64,
                        "physical_bytes": 8192,
                        "physical_read_bytes": 8192,
                        "pages_touched": 2,
                        "read_operations": 2,
                        "media_waves": 2,
                        "service_ns": 120.0,
                        "energy_pj": 3.0,
                        "profile_evidence": "analytical",
                        "background_work": {"gc": "unknown"},
                    },
                }
            }
        },
    )
    return _summarize_nand_task_traffic(
        (task,), resource_owners={"component.hbf.read": "hbf.owner"}
    )


def test_nand_ledger_keeps_direct_memory_structure_and_owner():
    ledger = _ledger()
    assert ledger["logical_bytes"] == 64
    assert ledger["physical_bytes"] == 8192
    assert ledger["operation_counts"] == {"read": 1}
    assert ledger["resource_totals"]["component.hbf.read"]["owner"] == "hbf.owner"
    assert ledger["background_work"] == [{"gc": "unknown"}]


def test_static_report_exposes_storage_ledger_for_each_retention_policy():
    scenario = build_reference_scenario()
    scenario = replace(
        scenario,
        workload=replace(
            scenario.workload,
            scheduler=replace(scenario.workload.scheduler, mode="static"),
        ),
    )
    for policy in ("exact", "streaming", "aggregate"):
        result = run_scenario(scenario, retention_policy=policy)
        result = replace(
            result,
            execution=replace(result.execution, storage_traffic=_ledger()),
        )
        assert report_dict(result)["storage_traffic"]["physical_bytes"] == 8192


def test_online_report_merges_bounded_batch_storage_ledger():
    result = run_scenario(build_reference_scenario(), retention_policy="aggregate")
    first = result.serving.batches[0]
    metadata = dict(first.cost.metadata)
    metadata["storage_traffic"] = _ledger()
    batch = replace(first, cost=replace(first.cost, metadata=metadata))
    serving = replace(result.serving, batches=(batch,) + tuple(result.serving.batches[1:]))
    report = report_dict(replace(result, serving=serving))
    assert report["storage_traffic"]["logical_bytes"] == 64
    assert report["storage_traffic"]["resource_totals"]["component.hbf.read"]["owner"] == "hbf.owner"
