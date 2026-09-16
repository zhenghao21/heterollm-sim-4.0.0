"""Bounded synthetic resource probes; no model loading, simulation or GPU execution."""
from dataclasses import replace
import pytest
from heterollm_sim.cost_models import GemmWorkload, HBMProfile, estimate_gpu_gemm
from heterollm_sim.mmvq_work import MMVQSourceContract, SOURCE_SHA256, derive_mmvq_work
from heterollm_sim.mmvq_issue_bound import MMVQIssueContract, source_issue_contract
from test_cost_models import gpu_profile, cache_hierarchy
from test_mmq_tail_costs import scenario as mmq_workload

MODE = "nominal_bandwidth_analytical_fallback"


def fixture(fmt="Q4_K", m=1, k=2048, n=2048, issue=False):
    source = MMVQSourceContract(1200, 1200, 32, dict(SOURCE_SHA256), "a"*64, True, False, True)
    work = derive_mmvq_work(m=m, k=k, n=n, weight_format=fmt, contract=source, allow_k_formats=True)
    gpu = gpu_profile(kernel_launch_ns=123, cache=cache_hierarchy(capacity_bytes=4096, bandwidth_gb_s=10000, hit_latency_ns=17, resource_id="gpu.l1"))
    gpu = replace(gpu, tensor_core=replace(gpu.tensor_core, sm_count=84, tensor_cores_per_sm=4), occupancy=.75, attainable_efficiency=.8)
    memory = HBMProfile(bandwidth_gb_s=900, efficiency=.8, energy_pj_per_byte=3, resource_id="gpu.hbm")
    load = GemmWorkload(m,k,n,activation_bits=8,weight_bits={"Q4_K":4,"Q6_K":6,"Q5_0":5,"Q8_0":8}[fmt],output_bits=32,
        packed_weight_formats=(fmt,), weight_storage_bytes=work.logical_weight_bytes,
        activation_storage_bytes=work.consumer_q8_1_unique_bytes,output_storage_bytes=work.output_f32_bytes,mmvq_work=work,
        packed_weight_transform_operations=m*k*n,
        mmvq_issue_contract=MMVQIssueContract.from_mapping(source_issue_contract(runtime_binary_sha256="a"*64,sm_count=84)) if issue else None)
    return gpu,memory,load


def demands(estimate):
    return {(p.name,d.resource_id):d for p in estimate.phases for d in p.demands}


@pytest.mark.parametrize("issue",[False,True])
@pytest.mark.parametrize("fmt,m,k,n",[("Q4_K",1,1792,2048),("Q4_K",1,2048,2048),("Q4_K",1,2304,2048),
    ("Q4_K",2,2048,2048),("Q4_K",4,2048,2048),("Q4_K",5,2048,2048),
    ("Q6_K",1,2048,2048),("Q5_0",1,896,896),("Q8_0",1,1024,16)])
def test_only_hbm_factor_changes_under_explicit_nominal_mode(fmt,m,k,n,issue):
    gpu,memory,load=fixture(fmt,m,k,n,issue)
    default=estimate_gpu_gemm(gpu,memory,load)
    assert default==estimate_gpu_gemm(gpu,memory,load,mmvq_hbm_mode="legacy_mma_output_wave")
    changed=estimate_gpu_gemm(gpu,memory,load,mmvq_hbm_mode=MODE)
    before,after=demands(default),demands(changed)
    assert set(before)==set(after)
    for key,d in before.items():
        if key[1]!=memory.resource_id:assert after[key]==d
        else:
            assert replace(after[key],service_ns=d.service_ns)==d
            assert after[key].service_ns==pytest.approx(d.service_ns*default.metadata["hbm_bandwidth"]["output_tile_wave_utilization"])
    assert default.useful_ops==changed.useful_ops and default.metadata["compute_service_ns"]==changed.metadata["compute_service_ns"]
    assert default.metadata["mmvq_source_work"]==changed.metadata["mmvq_source_work"]
    if issue:
        assert changed.metadata["overall_timing_completeness"]=="partial_with_nominal_HBM_analytical_fallback"
        assert default.metadata["overall_timing_completeness"]=="partial_with_legacy_HBM_fallback"
    bw=changed.metadata["hbm_bandwidth"]
    assert bw["model"]==MODE and bw["applied_hbm_shape_factor"]==1
    assert bw["shape_effective_hbm_bandwidth_gb_s"]==memory.effective_bandwidth_gb_s
    assert bw["bandwidth_saturation_proven"] is False and bw["measured_bandwidth_or_occupancy"] is False
    assert bw["output_wave_utilization_applied_to_hbm"] is False
    assert bw["legacy_mma_output_wave_diagnostic"] and not bw["legacy_mma_diagnostic_applied_to_hbm"]
    assert bw["validation_status"]=="unvalidated_nominal_bandwidth_assumption"
    assert bw["validated_kernel_families"]==bw["validated_hardware"]==()
    assert bw["source_ctas_below_sm_count"]==(load.mmvq_work.cta_count<84)


@pytest.mark.parametrize("n",[32,84,168])
def test_cta_count_never_becomes_memory_saturation_proof(n):
    gpu,memory,load=fixture(n=n)
    bw=estimate_gpu_gemm(gpu,memory,load,mmvq_hbm_mode=MODE).metadata["hbm_bandwidth"]
    assert bw["mmvq_nominal_mode_applied"] and not bw["bandwidth_saturation_proven"]
    assert bw["shape_effective_hbm_bandwidth_gb_s"]==memory.effective_bandwidth_gb_s


@pytest.mark.parametrize("change,reason",[("missing","missing_typed"),("geometry","noncanonical"),("bytes","physical_weight"),("fusion","fused_epilogue")])
def test_unqualified_work_remains_legacy_and_denominator_is_not_rejected(change,reason):
    gpu,memory,load=fixture()
    if change=="missing":load=replace(load,mmvq_work=None)
    elif change=="geometry":load=replace(load,mmvq_work=replace(load.mmvq_work,grid=(1,1,1)))
    elif change=="bytes":load=replace(load,weight_storage_bytes=load.weight_bytes+144)
    else:load=replace(load,epilogue_operations=1,epilogue_name="synthetic_epilogue")
    before=estimate_gpu_gemm(gpu,memory,load);after=estimate_gpu_gemm(gpu,memory,load,mmvq_hbm_mode=MODE)
    assert before.phases[0].demands==after.phases[0].demands and demands(before)==demands(after)
    bw=after.metadata["hbm_bandwidth"]
    assert not bw["mmvq_nominal_mode_applied"] and reason in bw["mmvq_nominal_mode_reason"]
    assert bw["model"]=="mma_output_tile_wave_proxy_v1" and bw["output_wave_utilization_applied_to_hbm"]


def test_non_mmvq_plain_and_source_mmq_keep_every_resource_demand():
    gpu,memory,load=fixture();plain=replace(load,mmvq_work=None,packed_weight_formats=(),packed_weight_transform_operations=0)
    gpu2,_,mmq=mmq_workload(896)
    for g,w in [(gpu,plain),(gpu2,mmq)]:
        before=estimate_gpu_gemm(g,memory,w);after=estimate_gpu_gemm(g,memory,w,mmvq_hbm_mode=MODE)
        assert demands(before)==demands(after) and before.service_ns==after.service_ns
        assert after.metadata["hbm_bandwidth"]["mmvq_nominal_mode_applied"] is False


def test_cache_payload_accounting_preserved():
    gpu,memory,load=fixture()
    before=estimate_gpu_gemm(gpu,memory,load);after=estimate_gpu_gemm(gpu,memory,load,mmvq_hbm_mode=MODE)
    a,b=dict(before.metadata["cache"]),dict(after.metadata["cache"])
    a.pop("backing_service_ns");b.pop("backing_service_ns")
    assert a==b and a["levels"]
    backing=demands(after)["gpu_gemm",memory.resource_id]
    assert backing.bytes_moved==demands(before)["gpu_gemm",memory.resource_id].bytes_moved
    assert sum(1 for p in before.phases if p.name=="kernel_launch")==sum(1 for p in after.phases if p.name=="kernel_launch")==1


def test_unknown_mode_is_not_silently_enabled():
    with pytest.raises(ValueError,match="mmvq_hbm_mode"):estimate_gpu_gemm(*fixture(),mmvq_hbm_mode="measured")


@pytest.mark.parametrize("change",[{"warps_per_cta":4.0},{"small_k":0},{"grid":(2048.0,1,1)},{"block":(32,4.0,1)},
    {"loop_iterations_by_thread":(1.0,)*128}])
def test_nominal_mode_requires_exact_source_geometry_types(change):
    gpu,memory,load=fixture()
    load=replace(load,mmvq_work=replace(load.mmvq_work,**change))
    before=estimate_gpu_gemm(gpu,memory,load);after=estimate_gpu_gemm(gpu,memory,load,mmvq_hbm_mode=MODE)
    assert demands(before)==demands(after)
    assert after.metadata["hbm_bandwidth"]["mmvq_nominal_mode_applied"] is False


def test_gpu_profile_entry_forwards_the_explicit_mode():
    gpu,memory,load=fixture()
    assert gpu.estimate_gemm(memory,load)==estimate_gpu_gemm(gpu,memory,load)
    assert gpu.estimate_gemm(memory,load,mmvq_hbm_mode=MODE)==estimate_gpu_gemm(gpu,memory,load,mmvq_hbm_mode=MODE)
