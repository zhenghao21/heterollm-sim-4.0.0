from types import SimpleNamespace
from heterollm_sim.kernel_query_ledger import summarize_kernel_queries

def task(ident="a", k=896, phase="gpu_gemm", geometry=True):
    metadata={"phase":phase,"target_component":"gpu0","kernel_main_consumer_storage_bytes":36*k//32, "mmq_source_work":{"status":"mmvq_precedes_mmq"}}
    if geometry:
        metadata["kernel_query_geometry"]={"m":1,"n":128,"k_logical":k,"weight_formats":["Q5_0"],
          "activation_storage_bytes":4*k,"output_storage_bytes":512,"accumulator_bits":32,
          "model_weight_read":True,"rhs_is_activation":False}
    return SimpleNamespace(task_id=ident, metadata=metadata)

def test_k_is_part_of_key_and_repeated_tasks_are_deduplicated():
    a=task(); result=summarize_kernel_queries([a,a,task("b",1024),task("c"),task("launch",phase="kernel_launch")])
    assert result["gpu_gemm_tasks"]==3 and result["represented_tasks"]==3
    assert len(result["signatures"])==2
    assert sorted(r["task_count"] for r in result["signatures"])==[1,2]
    assert all(r["key"]["calibration_eligible"] is False for r in result["signatures"])
    assert all(r["key"]["k_executed"] is None for r in result["signatures"])

def test_missing_geometry_retains_denominator():
    result=summarize_kernel_queries([task(),task("missing",geometry=False)])
    assert result["gpu_gemm_tasks"]==2 and result["unrepresented_tasks"]==1
    assert result["complete_geometry"] is False

def test_no_clock_or_model_index_in_query():
    value=task();value.metadata.update(model_name="secret",native_timing_ns=99,prompt_fingerprint="answer")
    result=summarize_kernel_queries([value]);key=result["signatures"][0]["key"]
    assert not any(name in key for name in ("model_name","native_timing_ns","prompt_fingerprint"))
    assert key["cache_state"]=="unknown" and key["native_dispatch_proven"] is False

def test_prediction_report_keeps_complete_compact_ledger():
    from tools.predict_stable_native_dataset import retained_dispatch_summary
    ledger=summarize_kernel_queries([task()])
    batch=SimpleNamespace(cost=SimpleNamespace(metadata={"gpu_invocations":{"gpu_gemm_tasks":1,"kernel_query_ledger":ledger}}))
    result=SimpleNamespace(serving=SimpleNamespace(batches=(batch,batch),scheduler_metrics=SimpleNamespace(total_batches=2)))
    result=retained_dispatch_summary(result)["gpu_invocations"]["kernel_query_ledger"]
    assert result["represented_tasks"]==2 and result["complete"] is True
    assert result["signatures"][0]["task_count"]==2


def test_same_identity_conflicting_shape_is_rejected():
    import pytest
    with pytest.raises(ValueError,match="conflicting geometry"):
        summarize_kernel_queries([task(),task(k=1024)])

def test_incomplete_storage_is_not_complete_and_logical_consumer_are_distinct():
    good=task();bad=task("b");del bad.metadata["kernel_query_geometry"]["weight_formats"]
    result=summarize_kernel_queries([good,bad])
    assert result["complete_geometry"] is False and result["unrepresented_tasks"]==1
    key=result["signatures"][0]["key"]
    assert key["logical_input_storage_bytes"]==3584
    assert key["main_consumer_storage_bytes"]==1008
