"""A deferred physical preview retains DAG ownership, not executed timings."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hbf_hbm_experiments import scenario
from heterollm_sim.planner import TopologyAwareBatchCostProvider
from heterollm_sim.reporting import run_scenario
from heterollm_sim.serving import _OnlineRuntime, _execution_stages_from_metadata
from test_hbf_host_staging_integration import strict_staging_scenario


@pytest.mark.parametrize("medium", ["HBM", "HBF"])
def test_deferred_default_preserves_live_metrics_and_host_task_ownership(medium):
    configured = (scenario(prompt=4, count=1, output=2) if medium == "HBM"
                  else strict_staging_scenario(prompt=4, output=2, all_hbf=True, capacity=65536))
    before = run_scenario(configured, retention_policy="aggregate",
        batch_lowerer=TopologyAwareBatchCostProvider(configured, defer_physical_execution=False))
    after = run_scenario(configured, retention_policy="aggregate")
    assert after.serving.request_metrics == before.serving.request_metrics
    expected_source = "authored_physical_task_dag" if medium == "HBM" else "deferred_physical_task_dag"
    for old, new in zip(before.serving.batches, after.serving.batches, strict=True):
        assert new.cost.metadata["execution_stage_source"] == expected_source
        assert new.cost.metadata["physical_execution_scope"] == "persistent_live_kernel"
        assert new.cost.metadata["preview_physical_cost_scope"] == "deferred_to_live_persistent_runtime"
        assert new.cost.metadata["execution_stages_include_host_orchestration"] is True
        for key in ("resource_busy_ns", "dram_traffic", "storage_traffic", "host_orchestration_ns", "device_execution_ns"):
            assert new.cost.metadata[key] == old.cost.metadata[key]


def test_authored_physical_dag_cannot_claim_the_preview_duration_is_executed():
    metadata = {"execution_stage_source": "authored_physical_task_dag", "execution_stage_makespan_ns": 10.}
    assert _OnlineRuntime._trusted_execution_stage_duration(metadata,
        host_ns=0., device_ns=10., extension_ns=0.) is None
    metadata["execution_stage_source"] = "executed_task_dag_kernel_timeline"
    assert _OnlineRuntime._trusted_execution_stage_duration(metadata,
        host_ns=0., device_ns=10., extension_ns=0.) == 10.


def test_authored_dag_requires_actual_tasks_instead_of_an_opaque_service_proxy():
    stages, reason = _execution_stages_from_metadata({
        "execution_stage_source": "authored_physical_task_dag",
        "execution_stages": ({"stage_id": "stage", "component_id": "gpu0", "request_ids": ("r0",),
                              "service_ns": 10., "dependencies": ()},),
    })
    assert not stages
    assert reason == "execution stage is missing execution_tasks"
