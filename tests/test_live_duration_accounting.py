from heterollm_sim.planner import TopologyAwareBatchCostProvider
from heterollm_sim.reporting import report_dict, run_scenario
from test_embedding_traffic_audit import embedding_scenario


def test_live_duration_does_not_recharge_preview_stage_envelope_gap(monkeypatch):
    scenario = embedding_scenario()
    baseline = report_dict(run_scenario(scenario, retention_policy="aggregate"))
    original = TopologyAwareBatchCostProvider.estimate

    def changed_preview(self, scenario, cohort):
        cost = original(self, scenario, cohort)
        metadata = dict(cost["metadata"])
        # Stage-only envelopes do not include all shared-resource waits.
        # Changing this planning estimate must not add work to the raw DAG.
        metadata["execution_stage_makespan_ns"] *= 0.5
        return {**cost, "metadata": metadata}

    monkeypatch.setattr(TopologyAwareBatchCostProvider, "estimate", changed_preview)
    actual = report_dict(run_scenario(scenario, retention_policy="aggregate"))
    assert actual["summary"]["physical_live_batch_count"] == 3
    assert actual["summary"] == baseline["summary"]
    assert actual["requests"] == baseline["requests"]
    for report in (baseline, actual):
        assert not any(stage["stage_id"] == "runtime.device_extension"
                       for batch in report["batch_history"]
                       for stage in batch["cost"]["metadata"]["execution_stage_schedule"])
