"""Synthetic source/work/profile checks only; no GPU, model or target timing."""
from dataclasses import replace
import math
import pytest

from heterollm_sim import planner
from heterollm_sim.cost_models import estimate_gpu_gemm
from heterollm_sim.mmvq_work import derive_mmvq_work, UnsupportedMMVQ
from heterollm_sim.mmvq_issue_bound import (MMVQIssueContract, source_issue_contract,
    derive_issue_bound, issue_counts, HARDWARE_DOCUMENT_SHA256)
from tests.test_mmvq_work import contract as kernel_contract
from tests.test_mmvq_mechanism import _qualified_mmvq_case
from tests.test_mmq_planner import scenario as base_scenario

FLAG = "llama_cpp_mmvq_vector_issue_bound"
KEY = "llama_cpp_mmvq_vector_issue_contract"


def work(fmt="Q5_0", m=4, k=1024, n=256):
    return derive_mmvq_work(m=m,k=k,n=n,weight_format=fmt,
        contract=kernel_contract(),allow_k_formats=True)


def cap(sm=84, **change):
    raw=source_issue_contract(runtime_binary_sha256="a"*64,sm_count=sm)
    raw.update(change)
    return MMVQIssueContract.from_mapping(raw)


def scenario(fmt="Q5_0", tokens=4, enabled=True, declaration=True, **contract_changes):
    base=base_scenario(tokens=tokens,weight_format=fmt)
    qualified=_qualified_mmvq_case()
    flags={**base.workload.metadata,**qualified.workload.metadata}
    if enabled is not None:flags[FLAG]=enabled
    raw=source_issue_contract(runtime_binary_sha256="a"*64,sm_count=84)
    raw.update(contract_changes)
    components=tuple(replace(c,metadata={**c.metadata,**({KEY:raw} if declaration else {})})
        if c.component_id=="gpu0" else c for c in qualified.hardware.components)
    return replace(base,hardware=replace(qualified.hardware,components=components),
        workload=replace(base.workload,metadata=flags))


def mains(schedule):
    return [t for t in schedule.tasks if t.metadata.get("phase")=="gpu_gemm"
        and t.metadata.get("mmvq_vector_issue_treatment",{}).get("status")=="applied_conditional_lower_bound"]


@pytest.mark.parametrize("fmt,max_m,qk,qi,vdr,block_bytes",[
    ("Q5_0",8,32,4,2,22),("Q8_0",8,32,8,2,34),
    ("Q4_K",5,256,32,2,144),("Q6_K",7,256,32,1,210)])
def test_source_instruction_counts_formats_geometry_and_units(fmt,max_m,qk,qi,vdr,block_bytes):
    for m in (1,2,4,max_m):
        for k in (qk,qk*3,qk*32):
            w=work(fmt,m,k)
            counts=issue_counts(w)
            assert (w.qk,w.qi,w.vdr,w.weight_block_bytes)==(qk,qi,vdr,block_bytes)
            assert counts["dp4a_weight_dependent_thread_calls"]==m*w.n*k//4
            assert counts["logical_math_operations"]==2*m*w.n*k
            assert counts["dp4a_weight_dependent_thread_calls"]*8==counts["logical_math_operations"]
            assert counts["dp4a_warp_issue_slots"]*32>=counts["dp4a_weight_dependent_thread_calls"]
            assert counts["dependency_latency_priced"] is False
            assert counts["longest_source_float_accumulator_updates"]==w.maximum_thread_k_iterations
            assert counts["independent_accumulators_per_thread"]==m*w.rows_per_cta
            assert w.logical_weight_bytes==w.n*k//qk*block_bytes
            assert counts["dp4a_correction_thread_expressions"]==(m*w.n*k//4 if fmt=="Q4_K" else 0)
    with pytest.raises(UnsupportedMMVQ):work(fmt,max_m+1,qk)


def test_partial_warps_cost_issue_slots_not_fractional_warps():
    w=work("Q5_0",1,32,64);c=issue_counts(w)
    assert w.active_k_threads_per_cta==2
    assert c["dp4a_warp_issue_slots"]>c["dp4a_weight_dependent_thread_calls"]/32
    assert c["dp4a_issue_slots_by_warp_per_cta"]==(16,0,0,0)
    b=derive_issue_bound(w,cap(),sm_count=84,frequency_ghz=2.)
    assert b["active_sm_upper_bound"]==16
    assert b["serial_warp_issue_floor_cycles"]==16
    assert b["lower_bound_sm_cycles"]==16 and b["service_ns"]==8
    assert b["native_instruction_mapping_proven"] is False
    assert b["dp4a_execution_throughput_known"] is False


@pytest.mark.parametrize("fmt,boundary",[("Q4_K",2048),("Q6_K",1024)])
def test_k_formats_strict_small_k_boundary_and_tail_fail_closed(fmt,boundary):
    assert work(fmt,1,boundary-256).small_k
    assert not work(fmt,1,boundary).small_k
    assert not work(fmt,2,256).small_k
    with pytest.raises(UnsupportedMMVQ):work(fmt,1,257)
    with pytest.raises(UnsupportedMMVQ):work(fmt,2,1024,3)
    with pytest.raises(UnsupportedMMVQ):
        derive_mmvq_work(m=1,k=1024,n=256,weight_format=fmt,contract=kernel_contract())


def test_conditioned_clock_and_hardware_binding_without_fitted_rate():
    w=work("Q6_K")
    a=derive_issue_bound(w,cap(),sm_count=84,frequency_ghz=1.)
    b=derive_issue_bound(w,cap(),sm_count=84,frequency_ghz=2.)
    assert a["service_ns"]==2*b["service_ns"]
    assert "not_wall_time" in a["clock_condition"]
    assert a["occupancy_efficiency_discount_applied"] is False
    assert a["source_contract"]["hardware_document_sha256"]==HARDWARE_DOCUMENT_SHA256
    for kwargs in ({"sm_count":80,"frequency_ghz":2.},{"sm_count":84,"frequency_ghz":float("nan")}):
        with pytest.raises(UnsupportedMMVQ):derive_issue_bound(w,cap(),**kwargs)
    with pytest.raises(UnsupportedMMVQ):issue_counts(replace(w,loop_iterations_by_thread=(1,)))


@pytest.mark.parametrize("change",[{"hardware_document_sha256":"0"*64},{"compute_capability":1000},
    {"dispatch_partitions_per_sm":8},{"threads_per_partition_cycle":128},
    {"clock_condition":"actual_wall_time_upper_bound"},{"instruction_model":"measured_dp4a_throughput"},
    {"warp_size":64},{"sm_count":True}])
def test_unverified_hardware_rates_and_conditions_rejected(change):
    with pytest.raises(UnsupportedMMVQ):cap(**change)


@pytest.mark.parametrize("fmt",["Q5_0","Q8_0","Q4_K","Q6_K"])
def test_enabled_replaces_tensor_dot_no_duplicate_unpack_and_keeps_kernel_count(fmt):
    enabled=planner.compile_scenario(scenario(fmt))
    old=planner.compile_scenario(scenario(fmt,enabled=False))
    selected=mains(enabled)
    assert selected
    assert [t.name for t in enabled.tasks]==[t.name for t in old.tasks]
    by_name={t.name:t for t in old.tasks}
    for t in selected:
        baseline=by_name[t.name];cost=t.metadata["cost_model"]
        assert not any(d.resource_id.endswith("tensor_core") for d in t.demands)
        scalar=[d for d in t.demands if d.resource_id.endswith("scalar")]
        assert len(scalar)==1 and scalar[0].service_ns>0
        assert scalar[0].energy_pj==0 and cost["compute_energy_priced"] is False
        assert cost["mma_compute_priced"] is False and cost["issued_operations"] is None
        assert cost["unpack_cost_priced"] is False
        assert "fused_dequant_service_ns" not in cost
        assert scalar[0].service_ns==cost["mmvq_vector_issue_bound"]["service_ns"]
        assert cost["hbm_bandwidth"]["source_geometry_hbm_concurrency_applied"] is False
        assert "legacy" in cost["hbm_bandwidth"]["source_geometry_hbm_concurrency_reason"]
        assert "legacy_HBM" in cost["overall_timing_completeness"]
        assert cost["activation_bytes"]==baseline.metadata["cost_model"]["activation_bytes"]
        assert cost["weight_bytes"]==baseline.metadata["cost_model"]["weight_bytes"]
        assert cost["output_bytes"]==baseline.metadata["cost_model"]["output_bytes"]
        assert [(d.resource_id,d.service_ns,d.bytes_moved) for d in t.demands if d.bytes_moved] == [
            (d.resource_id,d.service_ns,d.bytes_moved) for d in baseline.demands if d.bytes_moved]
    conversions=lambda s:[(t.name,t.demands) for t in s.tasks if "activation_q8_1" in t.name]
    assert conversions(enabled)==conversions(old)
    assert sum(t.metadata.get("phase")=="kernel_launch" for t in enabled.tasks)==sum(
        t.metadata.get("phase")=="kernel_launch" for t in old.tasks)


def test_disabled_matches_missing_switch_even_with_contract_present():
    explicit=planner.compile_scenario(scenario(enabled=False))
    absent=planner.compile_scenario(scenario(enabled=None))
    assert explicit.tasks==absent.tasks
    assert not mains(explicit)


@pytest.mark.parametrize("change",[{"declaration":False},{"hardware_document_sha256":"0"*64},
    {"runtime_binary_sha256":"b"*64},{"sm_count":80},{"instruction_model":"tensor_mma"}])
def test_missing_or_mismatched_contract_falls_back_without_rate_guess(change):
    schedule=planner.compile_scenario(scenario(**change))
    assert not mains(schedule)
    audits=[t.metadata["mmvq_vector_issue_treatment"] for t in schedule.tasks
        if "mmvq_vector_issue_treatment" in t.metadata]
    assert audits and all(x["status"]=="uncovered" and x["reason"] for x in audits)


@pytest.mark.parametrize("fmt",["IQ4_XS","IQ3_S","Q5_K"])
def test_other_formats_keep_legacy_numeric_path_and_explicit_uncovered(fmt):
    a=planner.compile_scenario(scenario(fmt));b=planner.compile_scenario(scenario(fmt,enabled=False))
    assert not mains(a)
    assert [(t.name,t.demands) for t in a.tasks]==[(t.name,t.demands) for t in b.tasks]


def test_issue_cost_ignores_generic_scalar_rate_efficiency_and_occupancy():
    case=scenario("Q4_K")
    first=planner.compile_scenario(case)
    profiles={**case.component_profiles,"gpu":{key:replace(profile,scalar_lanes_per_sm=1,
        scalar_ops_per_cycle=.01,occupancy=.1,attainable_efficiency=.2)
        for key,profile in case.component_profiles["gpu"].items()}}
    second=planner.compile_scenario(replace(case,component_profiles=profiles))
    assert {t.name:t.metadata["cost_model"]["compute_service_ns"] for t in mains(first)}=={
        t.name:t.metadata["cost_model"]["compute_service_ns"] for t in mains(second)}


def test_non_boolean_switch_is_rejected_instead_of_silently_enabling():
    with pytest.raises(ValueError,match="explicit boolean"):
        planner.compile_scenario(scenario(enabled="true"))


def test_competing_unpack_partial_treatment_fails_closed():
    case=scenario("Q4_K")
    case=replace(case,hardware=replace(case.hardware,metadata={**case.hardware.metadata,
        "llama_cpp_mmvq_prmt_partial_contract":{"enabled":True}}))
    schedule=planner.compile_scenario(case)
    assert not mains(schedule)
    assert any(t.metadata.get("mmvq_vector_issue_treatment",{}).get("reason") ==
        "competing_PRMT_partial_cost_treatment" for t in schedule.tasks)


def test_direct_vector_cost_does_not_require_tensor_dtype_or_tensor_cycles():
    from unittest.mock import patch
    case=scenario("Q6_K")
    with patch.object(planner,"estimate_gpu_gemm",wraps=estimate_gpu_gemm) as estimator:
        planner.compile_scenario(case)
    args=next(call.args for call in estimator.call_args_list if call.args[2].mmvq_issue_contract is not None)
    gpu,hbm,workload=args
    original=estimate_gpu_gemm(gpu,hbm,workload)
    changed=replace(gpu,default_tensor_dtype="bf16",tensor_core=replace(gpu.tensor_core,cycles_per_mma=999.,
        supported_dtypes=("bf16",),dtype_throughput_scale={"bf16":0.5}))
    actual=estimate_gpu_gemm(changed,hbm,workload)
    assert actual.metadata["compute_service_ns"]==original.metadata["compute_service_ns"]
    assert all(d.resource_id != changed.tensor_core.resource_id for phase in actual.phases for d in phase.demands)
    assert actual.metadata["quantized_format_coverage"]=="source_MMVQ_conditional_issue_bound"
    assert actual.metadata["mmvq_source_work"]["cost_model_applied"] is True
