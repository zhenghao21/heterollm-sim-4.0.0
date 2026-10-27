from heterollm_sim.native_pipeline_sampling import compose_fifo_points


def test_pipeline_fifo_preserves_stage_order_and_refill_bubble():
    points = compose_fifo_points({0: 3.0, 1: 5.0}, {0: 2.0}, 3, arrivals=(0.0, 0.0, 20.0))
    stage1 = [row for row in points if row.stage == 1 and row.kind == "compute"]
    assert [row.start_ns for row in stage1] == [5.0, 10.0, 25.0]
    assert stage1[1].bubble_ns > 0


def test_pipeline_fifo_rejects_missing_activation_transfer():
    import pytest
    with pytest.raises(ValueError, match="transfer_missing"):
        compose_fifo_points({0: 3.0, 1: 5.0}, {}, 2)
