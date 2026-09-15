from types import SimpleNamespace
import pytest
from heterollm_sim.planner import summarize_tensor_storage


def test_count_physical_embedding_once_keep_capacity_and_route_bytes_separate():
    audit = dict(requested=True, qualified=True, selected_rows=2, selected_elements=64,
                 selected_weight_read_bytes=68, index_read_bytes=8, output_write_bytes=256,
                 rank_weight_capacity_bytes=3400)
    def item(phase, kind="embedding", **extras):
        return SimpleNamespace(metadata=dict(phase=phase,event_kind=kind,native_get_rows_storage=audit,**extras))
    tasks=[item("cpu_memory"), item("gpu_memory"), item("kernel_launch"),
           item("transfer","model_weight_transfer",weight_access_semantics="full_weight_staging_before_row_lookup",
                weight_read_invocation_id="one",rank_weight_capacity_bytes=3400),
           item("transfer","model_weight_transfer",weight_access_semantics="full_weight_staging_before_row_lookup",
                weight_read_invocation_id="one",rank_weight_capacity_bytes=3400)]
    s=summarize_tensor_storage(tasks)
    assert s["embedding_tasks"] == s["applied_tasks"] == 2
    assert s["selected_weight_read_bytes"] == 136
    assert s["index_read_bytes"] == 16
    assert s["output_write_bytes"] == 512
    assert s["rank_weight_capacity_bytes"] == 3400
    assert s["full_table_staging_bytes"] == 3400
    assert s["selected_rows"] == 4
    assert s["dequantization_compute_priced"] is False


def test_missing_qualified_bytes_cannot_become_zero_evidence():
    task=SimpleNamespace(metadata=dict(phase="cpu_memory",event_kind="embedding",
                                      native_get_rows_storage=dict(requested=True,qualified=True)))
    with pytest.raises(ValueError,match="lacks"):
        summarize_tensor_storage([task])
