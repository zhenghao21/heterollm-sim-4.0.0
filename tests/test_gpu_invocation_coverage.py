from types import SimpleNamespace
import pytest
from heterollm_sim.planner import summarize_gpu_invocations, summarize_mmq_source_work


def item(phase, stage=None, status="applied", name="matrix", mapped=True,
         ident="task", owner="request", dependencies=(), qualified=True):
    m={"phase":phase,"op_name":name,"target_component":"gpu0"}
    if mapped:
        m["gpu_native_invocation"]={"status":"conditional" if qualified else "uncovered",
            "applied":qualified,"reason":None if qualified else "source_tensor_storage_mismatch",
            "physical_weight_matrices":1,"native_dispatch_proven":False}
    if stage:m["mmq_source_work"]={"stage":stage,"status":status}
    return SimpleNamespace(metadata=m,task_id=ident,request_id=owner,dependencies=dependencies)


def test_matrix_counts_are_not_duplicated_by_launch_and_fixup_phases():
    ts=[item("kernel_launch","matrix",ident="ml"),item("gpu_gemm","matrix",ident="mm"),
        item("kernel_launch","conversion",name="convert",ident="cl"),
        item("gpu_elementwise","conversion",name="convert",ident="cc",dependencies=("cl",)),
        item("kernel_launch","fixup",name="fixup",ident="fl"),
        item("gpu_elementwise","fixup",name="fixup",ident="fc",dependencies=("fl",)),
        item("gpu_gemm","matrix",status="mmvq_precedes_mmq",name="small",ident="small"),
        item("gpu_gemm",mapped=False,name="dynamic",ident="dyn")]
    a=summarize_gpu_invocations(ts);b=summarize_mmq_source_work(ts)
    assert a["gpu_gemm_tasks"]==3 and a["applied_tasks"]==2
    assert a["native_dispatch_proven_tasks"]==0
    assert b["gpu_gemm_tasks"]==3 and b["applied_tasks"]==1
    assert b["conversion_tasks"]==b["fixup_tasks"]==1
    assert b["matrix_status_counts"]=={"applied":1,"mmvq_precedes_mmq":1}
    assert b["native_dispatch_proven_tasks"]==0
    assert summarize_mmq_source_work(ts+ts)==b


def test_conversion_without_identity_fails_instead_of_silently_undercounting():
    with pytest.raises(ValueError,match="operation identity"):
        summarize_mmq_source_work([item("kernel_launch","conversion",name="")])


def test_same_name_cross_request_and_repeated_calls_keep_distinct_identity():
    ts=[]
    for index,owner in enumerate(("r1","r2","r1")):
        launch="launch"+str(index);device="device"+str(index)
        ts.extend((item("kernel_launch","conversion",ident=launch,owner=owner),
            item("gpu_elementwise","conversion",ident=device,owner=owner,dependencies=(launch,))))
    # Separate launch-less calls with the same name cannot collapse either.
    ts.extend((item("gpu_elementwise","conversion",ident="bare1"),
               item("gpu_elementwise","conversion",ident="bare2",dependencies=("bare1",)),
               item("kernel_launch","fixup",ident="empty-fixup")))
    result=summarize_mmq_source_work(ts)
    assert result["conversion_tasks"]==5
    assert result["fixup_tasks"]==1


def test_failed_projection_label_is_uncovered_not_applied():
    r=summarize_gpu_invocations([item("gpu_gemm",qualified=False)])
    assert r["audited_tasks"]==1 and r["applied_tasks"]==0
    assert r["uncovered_tasks"]==1
    assert r["uncovered_reason_counts"]=={"source_tensor_storage_mismatch":1}
