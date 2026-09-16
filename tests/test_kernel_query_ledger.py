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


def qualified_task(ident="qualified"):
    from heterollm_sim.mmvq_work import MMVQSourceContract, SOURCE_SHA256, derive_mmvq_work
    value=task(ident,k=2048)
    work=derive_mmvq_work(m=1,k=2048,n=2048,weight_format="Q4_K",allow_k_formats=True,
        contract=MMVQSourceContract(1200,1200,32,dict(SOURCE_SHA256),"a"*64,True,False,True))
    value.metadata["kernel_query_geometry"].update(n=2048,weight_formats=["Q4_K"],output_storage_bytes=8192)
    value.metadata["mmvq_source_work"]={**work.to_metadata(),"status":"source_geometry_unpriced","stage":"matrix","execution_component":"gpu0"}
    return value


def test_source_derived_mmvq_k_is_logical_and_never_upgrades_native_qualification():
    import json
    value=qualified_task()
    for audit in [value.metadata["mmvq_source_work"],json.loads(json.dumps(value.metadata["mmvq_source_work"]))]:
        value.metadata["mmvq_source_work"]=audit
        result=summarize_kernel_queries([value]);key=result["signatures"][0]["key"]
        assert key["k_executed"]==key["k_logical"]==2048
        assert key["k_execution_source"]=="source_derived_mmvq_logical_K"
        assert not key["k_execution_native_proven"] and not key["native_dispatch_proven"] and not key["layout_proven"] and not key["calibration_eligible"]
        assert key["cache_state"]=="unknown" and result["represented_tasks"]==1


def test_incomplete_and_tampered_mmvq_proof_retain_null_k_without_dropping_tasks():
    import copy
    changes=[lambda a:a.pop("runtime_binary_sha256"),lambda a:a.pop("source_hashes"),lambda a:a.update(grid=(1,1,1)),
        lambda a:a.update(k=4096),lambda a:a.update(halve_iters=True),lambda a:a.update(native_dispatch_proven=True),
        lambda a:a.update(status="uncovered")]
    for change in changes:
        value=qualified_task();change(value.metadata["mmvq_source_work"])
        result=summarize_kernel_queries([value]);key=result["signatures"][0]["key"]
        assert key["k_executed"] is None and key["k_execution_source"]=="unknown"
        assert key["mmvq_k_execution_reason"] and result["represented_tasks"]==1 and result["complete_geometry"]
    value=qualified_task();value.metadata["kernel_main_consumer_storage_bytes"]+=36
    key=summarize_kernel_queries([value])["signatures"][0]["key"]
    assert key["k_executed"] is None and key["mmvq_k_execution_reason"]=="mmvq_source_storage_mismatch"


def test_source_proof_is_part_of_deduplication_and_mmq_k_is_unchanged():
    import copy,pytest
    one=qualified_task();two=copy.deepcopy(one);two.metadata["mmvq_source_work"]["k"]=4096
    with pytest.raises(ValueError,match="conflicting geometry"):summarize_kernel_queries([one,two])
    value=task();value.metadata["mmq_source_work"]={"status":"applied","k_execution":1024}
    key=summarize_kernel_queries([value])["signatures"][0]["key"]
    assert key["k_executed"]==1024 and key["predicted_family"]=="MMQ"


def test_mmvq_numeric_types_and_extra_layout_contradictions_fall_back():
    for key,invalid in [("warps_per_cta",4.0),("small_k",0),("grid",[2048.0,1,1]),
            ("has_ids",True),("channels",2),("samples",2),("layout","strided"),
            ("ordinary_contiguous_2d",False),("execution_component","cpu0")]:
        value=qualified_task();value.metadata["mmvq_source_work"][key]=invalid
        result=summarize_kernel_queries([value]);record=result["signatures"][0]["key"]
        assert record["k_executed"] is None and result["represented_tasks"]==1,(key,invalid)
    for key,invalid in [("activation_storage_bytes",4096),("accumulator_bits",16)]:
        value=qualified_task();value.metadata["kernel_query_geometry"][key]=invalid
        result=summarize_kernel_queries([value]);record=result["signatures"][0]["key"]
        assert record["k_executed"] is None and result["represented_tasks"]==1,(key,invalid)
