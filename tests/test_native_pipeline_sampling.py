from types import SimpleNamespace

from heterollm_sim.native_pipeline_sampling import (compose_fifo_points,
    select_representative_pipeline_tasks, stage_samples_from_window)


def test_pipeline_fifo_preserves_stage_order_and_refill_bubble():
    points = compose_fifo_points({0: 3.0, 1: 5.0}, {0: 2.0}, 3, arrivals=(0.0, 0.0, 20.0))
    stage1 = [row for row in points if row.stage == 1 and row.kind == "compute"]
    assert [row.start_ns for row in stage1] == [5.0, 10.0, 25.0]
    assert stage1[1].bubble_ns > 0


def test_pipeline_fifo_rejects_missing_activation_transfer():
    import pytest
    with pytest.raises(ValueError, match="transfer_missing"):
        compose_fifo_points({0: 3.0, 1: 5.0}, {}, 2)


def test_single_microbatch_embedded_activation_transfer_is_sampled_separately():
    stage0 = SimpleNamespace(stage_id="operator-invocation-group-0000.stage0000",
        execution_tasks=(SimpleNamespace(task_id="compute0", metadata={"layer_id": "layer-000"}),))
    stage1 = SimpleNamespace(stage_id="operator-invocation-group-0000.stage0001",
        execution_tasks=(
            SimpleNamespace(task_id="compute1", metadata={"layer_id": "layer-001"}),
            SimpleNamespace(task_id="transfer", metadata={
                "event_kind": "pipeline_activation", "source_stage": 0,
                "target_stage": 1}),
        ))
    specs = [SimpleNamespace(task_id="compute0", metadata={
        "execution_stage_id": stage0.stage_id, "layer_id": "layer-000"}),
        SimpleNamespace(task_id="compute1", metadata={
            "execution_stage_id": stage1.stage_id, "layer_id": "layer-001"}),
        SimpleNamespace(task_id="transfer", metadata={
            "execution_stage_id": stage1.stage_id, "event_kind": "pipeline_activation",
            "source_stage": 0, "target_stage": 1})]
    selected, descriptors = select_representative_pipeline_tasks((stage0, stage1), specs)
    assert "transfer" in selected
    transfer = next(row for row in descriptors if row["kind"] == "pipeline_activation_transfer")
    assert transfer["stage_id"].endswith(".pp0000.activation_transfer")
    samples = stage_samples_from_window({
        "layer_start_ns": {"layer-000": 0, "layer-001": 20},
        "layer_completion_ns": {"layer-000": 10, "layer-001": 30},
        "pipeline_transfer_start_ns": {transfer["stage_id"]: 10},
        "pipeline_transfer_completion_ns": {transfer["stage_id"]: 20},
    }, descriptors, cohort=0)
    assert any(row.kind == "pipeline_activation_transfer" and row.duration_ns == 10
               for row in samples)
