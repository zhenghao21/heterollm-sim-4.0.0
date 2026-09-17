"""Static/structural norm placement tests; no native model, GPU or timing data."""
from dataclasses import replace
import pytest
from heterollm_sim import planner
from heterollm_sim.ir import LayerSpec, RequestSpec, WorkloadSpec
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
from heterollm_sim.llama_scenario import apply_llama_runtime_config, llama_final_norm_static_binding
from heterollm_sim.gguf_parity import GGUFMetadata, GGUFTensor, build_model_from_gguf
from heterollm_sim.final_layer_output_selection import SOURCE_KEY, source_declaration
from tests.model_helpers import model_from_layer_specs


def scenario(ngl=0, offload=False, dtype="F32", fusion=True, conditions=None,
             source_bound=True, cpu_fusion=True, cpu_use_ref=False):
    base = build_reference_scenario()
    bits = {"F32": 32, "F16": 16, "BF16": 16}[dtype]
    metadata = {"final_norm_weight_binding": {"name": "output_norm.weight", "shape": [32],
        "type": dtype, "n_bytes": 32 * bits // 8, "offset": 4096}, "final_norm_epsilon": 1e-5}
    if source_bound: metadata[SOURCE_KEY] = source_declaration()
    layers = tuple(LayerSpec("layer%d" % i, "dense", hidden_size=32, intermediate_size=64,
        attention_heads=4, kv_heads=2, weight_bytes=65536) for i in range(2))
    model = model_from_layer_specs("norm-audit", layers, vocabulary_size=64,
        max_sequence_length=256, embedding_weight_bytes=8192, metadata=metadata,
        architecture="llama_decoder" if source_bound else "transformer")
    parallel = replace(base.placement.parallel, layer_to_stage={x.layer_id: 0 for x in layers},
        rank_mapping=tuple(replace(r, cim_component_id=None) for r in base.placement.parallel.rank_mapping))
    base = replace(base, model=model, placement=replace(base.placement, model_name=model.name, parallel=parallel),
        workload=WorkloadSpec("norm-audit", requests=(RequestSpec("r", 0, 64, 1),)))
    cfg = LlamaCppRuntimeConfig(gpu_layers=ngl, op_offload=offload, context=256, batch=64, ubatch=64)
    result = apply_llama_runtime_config(base, cfg)
    extra = {"llama_cpp_f32_hidden_storage": True,
        "llama_cpp_cuda_op_offload": {"status": "enabled" if offload else "disabled", "minimum_m": 32},
        "llama_cpp_gpu_native_invocations": {"fusion_enabled": fusion},
        "llama_cpp_cpu_final_norm_controls": {"fusion_enabled":cpu_fusion,"use_ref":cpu_use_ref}}
    if conditions is not None: extra["llama_cpp_final_norm_graph_conditions"] = conditions
    return replace(result, workload=replace(result.workload, metadata={**result.workload.metadata, **extra}))


def tail(case, source="cpu0", rows=1):
    with planner._compilation_scope(case):
        plan = planner._parallel_plan(case); rank = plan.tp_group(plan.pp_degree - 1, 0)[0]
        builder = planner._TaskBuilder(RequestSpec("norm", 0, rows, 1))
        start = builder.add("activation_ready", planner.TaskCategory.POLICY)
        builder.record_rank_value(start, rank.rank, source)
        ends = planner._compile_parallel_final_norm(builder, case, plan, planner._topology_router(case),
            "prefill", (start,), token_batch=rows)
        return tuple(builder.tasks), ends[rank.rank]


def mains(tasks):
    return [t for t in tasks if "final_norm_work" in t.metadata
            and t.metadata.get("phase") not in ("kernel_launch", "cpu_dispatch")]


def test_cpu_only_final_norm_has_no_gpu_or_transfer():
    tasks, _ = tail(scenario())
    assert [t.metadata["final_norm_work"]["native_op"] for t in mains(tasks)] == ["RMS_NORM+MUL"]
    assert all(t.metadata.get("target_component") == "cpu0" for t in mains(tasks))
    assert not any("gpu" in d.resource_id or "pcie" in d.resource_id for t in tasks for d in t.demands)


@pytest.mark.parametrize("ngl,expected", [(0,"cpu"),(1,"rank_gpu"),(2,"rank_gpu"),(3,"rank_gpu"),(-1,"rank_gpu")])
def test_locked_output_inclusive_tail_candidates(ngl, expected):
    case=scenario(ngl)
    assert llama_final_norm_static_binding(case,case.llama_cpp_config)["output_device_candidate"] == expected
    tasks,_=tail(case)
    assert all(t.metadata["target_component"] == ("cpu0" if ngl==0 else "gpu0") for t in mains(tasks))


@pytest.mark.parametrize("rows,gpu", [(31,False),(32,True),(33,True)])
def test_weight_mul_offload_uses_norm_rows_not_head_rows(rows,gpu):
    tasks,_=tail(scenario(offload=True),rows=rows)
    proof=mains(tasks)[-1].metadata["final_norm_placement"]
    assert proof["norm_rows"]==rows and proof["weight_scale_offload_applied"] is gpu
    assert proof["scale_component"] == ("gpu0" if gpu else "cpu0")


def test_gpu_activation_cpu_weight_retains_true_split_and_one_gpu_kernel():
    tasks,end=tail(scenario(),source="gpu0",rows=2)
    ops=mains(tasks)
    assert [(t.metadata["final_norm_work"]["native_op"], t.metadata["target_component"]) for t in ops] == [("RMS_NORM","gpu0"),("MUL","cpu0")]
    assert sum(t.metadata.get("phase")=="kernel_launch" for t in tasks)==1
    moves=[t for t in tasks if t.metadata.get("transfer_payload")=="normalized_activation"]
    assert moves and all(t.metadata["input_bytes"]==256 for t in moves)
    assert end==ops[-1].task_id


def test_fused_gpu_has_one_dispatch_no_external_rms_temporary():
    tasks,_=tail(scenario(-1),source="gpu0",rows=2)
    ops=mains(tasks);assert len(ops)==1
    w=ops[0].metadata["final_norm_work"]
    assert w=={**w,"native_op":"RMS_NORM+MUL","logical_operations":258,
               "logical_read_bytes":768,"logical_write_bytes":256,"transcendental_operations":2,
               "external_reduction_temporary_bytes":0,"physical_dispatches":1}
    assert sum(t.metadata.get("phase")=="kernel_launch" for t in tasks)==1


@pytest.mark.parametrize("dtype,fusion", [("F16",True),("F32",False)])
def test_nonfused_gpu_is_complete_rms_plus_mul_two_dispatches(dtype,fusion):
    tasks,_=tail(scenario(-1,dtype=dtype,fusion=fusion),source="gpu0",rows=2)
    ops=mains(tasks)
    assert len(ops)==2 and sum(t.metadata.get("phase")=="kernel_launch" for t in tasks)==2
    assert sum(t.metadata["final_norm_work"]["logical_operations"] for t in ops)==258
    assert [t.metadata["final_norm_work"]["logical_write_bytes"] for t in ops]==[256,256]
    assert all(t.metadata["final_norm_work"]["external_reduction_temporary_bytes"]==0 for t in ops)


def test_host_weight_can_move_into_fused_cuda_split_once():
    tasks,_=tail(scenario(offload=True),rows=32)
    assert len(mains(tasks))==1
    moves=[t for t in tasks if t.metadata.get("transfer_payload")=="output_norm.weight"]
    assert moves and all(t.metadata["input_bytes"]==128 for t in moves)
    assert not any(t.metadata.get("transfer_payload")=="normalized_activation" for t in tasks)


def test_unknown_fusion_does_not_add_unproven_launch():
    tasks,_=tail(scenario(-1,fusion=None),source="gpu0")
    p=mains(tasks)[0].metadata["final_norm_placement"]
    assert p["fusion_status"]=="conditional_fused_lower_envelope"
    assert p["unpriced_additional_launch_if_unfused"]==1
    assert sum(t.metadata.get("phase")=="kernel_launch" for t in tasks)==1


@pytest.mark.parametrize("conditions", [{"adjacent":False},{"rms_single_use":False},
    {"rms_is_view":True},{"rms_is_output":True},{"weight_contiguous_rows":False},
    {"shape_connection_matches":False}])
def test_fusion_requires_graph_and_layout_conditions(conditions):
    tasks,_=tail(scenario(-1,conditions=conditions),source="gpu0")
    assert len(mains(tasks))==2


def test_authored_coarse_placement_is_preserved():
    case=scenario(-1);case=replace(case,placement=replace(case.placement,
        op_to_component={**case.placement.op_to_component,"final_norm":"cpu0"}))
    tasks,_=tail(case,source="cpu0")
    assert not mains(tasks)
    assert all(t.metadata.get("target_component")!="gpu0" for t in tasks)


def test_norm_weight_override_is_not_inherited_from_lm_head():
    case=scenario(-1)
    case=replace(case,placement=replace(case.placement,tensor_to_component={
        **case.placement.tensor_to_component,"output_norm.weight":"hostmem0"}))
    tasks,_=tail(case,source="gpu0")
    assert [t.metadata["target_component"] for t in mains(tasks)]==["gpu0","cpu0"]


def test_static_norm_width_and_bytes_fail_closed():
    case=scenario();model=case.model
    for shape,nbytes in (([16],64),([32],64)):
        meta={"final_norm_weight_binding":{"name":"output_norm.weight","shape":shape,"type":"F32","n_bytes":nbytes}}
        bad=replace(case,model=replace(model,metadata=meta,graph=replace(model.graph,attributes={**model.graph.attributes,"metadata":meta})))
        with pytest.raises(ValueError):llama_final_norm_static_binding(bad,bad.llama_cpp_config)


def gguf_fixture_model(norm_dtype="F16"):
    specs=[("token_embd.weight",(32,64)),("output.weight",(32,64)),
        ("blk.0.attn_q.weight",(32,32)),("blk.0.attn_k.weight",(32,16)),
        ("blk.0.attn_v.weight",(32,16)),("blk.0.attn_output.weight",(32,32)),
        ("blk.0.ffn_gate.weight",(32,64)),("blk.0.ffn_up.weight",(32,64)),("blk.0.ffn_down.weight",(64,32))]
    ts=[GGUFTensor(n,sh,0,"F32",1,sh[0]*sh[1]*4,0) for n,sh in specs]
    bits = 32 if norm_dtype == "F32" else 16
    ts.append(GGUFTensor("output_norm.weight",(32,),0 if bits == 32 else 1,norm_dtype,1,32 * bits // 8,777))
    data=GGUFMetadata("synthetic", "a"*64,3,len(ts),0,"qwen2",1,32,4,2,64,256,"F32",0,
        {"qwen2.attention.layer_norm_rms_epsilon":1e-5},tuple(ts))
    return build_model_from_gguf(data)


def test_gguf_keeps_norm_tensor_own_type_shape_and_bytes():
    model=gguf_fixture_model()
    meta=model.graph.attributes["metadata"]
    assert meta["gguf_output_norm_binding"]=={"name":"output_norm.weight","shape":[32],"type":"F16","n_bytes":64,"offset":777}
    assert meta["gguf_norm_epsilon"]==1e-5


@pytest.mark.parametrize("rows,fused", [(1,True),(2,False)])
def test_rms_as_mul_right_operand_requires_other_shape_match(rows,fused):
    tasks,_=tail(scenario(-1,conditions={"rms_is_mul_left_operand":False}),source="gpu0",rows=rows)
    assert len(mains(tasks)) == (1 if fused else 2)


def test_foreign_weight_same_device_fusion_keeps_split_uncertainty():
    tasks,_=tail(scenario(offload=True),rows=32)
    audit=mains(tasks)[0].metadata["final_norm_placement"]
    assert audit["fusion_conditions"]["same_split"] is None
    assert audit["fusion_status"]=="conditional_fused_lower_envelope"


@pytest.mark.parametrize("architecture,rows", [("llama_decoder",1),("qwen2_decoder",1),("qwen3_5_hybrid_transformer",64)])
def test_output_selection_keeps_source_defined_norm_rows(architecture,rows):
    from tests.test_final_layer_output_selection_planner import _scenario, _cohort
    case=_scenario(architecture)
    metadata={**case.model.graph.attributes.get("metadata",{}),
        "final_norm_weight_binding":{"name":"output_norm.weight","shape":[32],"type":"F32","n_bytes":128},
        "final_norm_epsilon":1e-5}
    case=replace(case,model=replace(case.model,metadata=metadata,
        graph=replace(case.model.graph,attributes={**case.model.graph.attributes,"metadata":metadata})),
        llama_cpp_config=LlamaCppRuntimeConfig(gpu_layers=-1,context=256,batch=64,ubatch=64),
        workload=replace(case.workload,metadata={**case.workload.metadata,
            "llama_cpp_gpu_native_invocations":{"fusion_enabled":True}}))
    schedule=planner.compile_serving_cohort_schedule(case,_cohort(64,1))
    norms=mains(schedule.tasks)
    assert len(norms)==1
    assert norms[0].metadata["final_norm_placement"]["norm_rows"]==rows
    assert norms[0].metadata["final_layer_output_selection"]["final_norm_rows"]==rows


@pytest.mark.parametrize("disabled,reference", [(False,False),(True,True)])
def test_cpu_unfused_rms_counts_memcpy_and_inplace_scale(disabled,reference):
    # Either disabled fusion or a reference plan prevents the CPU fused branch.
    case=scenario(cpu_fusion=disabled,cpu_use_ref=reference)
    tasks,_=tail(case,rows=2)
    ops=mains(tasks)
    assert [t.metadata["final_norm_work"]["native_op"] for t in ops]==["RMS_NORM","MUL"]
    rms,mul=[t.metadata["final_norm_work"] for t in ops]
    assert rms["logical_read_bytes"]==3*256 and rms["logical_write_bytes"]==2*256
    assert mul["logical_read_bytes"]==256+256 and mul["logical_write_bytes"]==256
    assert sum(t.metadata.get("phase")=="cpu_dispatch" for t in tasks)==2
    assert ops[0].metadata["final_norm_placement"]["fusion_status"]=="separate_native_operations"


def test_cpu_fused_rms_mul_has_two_activation_passes_one_output_write():
    tasks,_=tail(scenario(),rows=2)
    ops=mains(tasks);assert len(ops)==1
    work=ops[0].metadata["final_norm_work"]
    assert work["native_op"]=="RMS_NORM+MUL"
    assert work["logical_read_bytes"]==768 and work["logical_write_bytes"]==256
    assert work["logical_operations"]==260  # CPU reciprocal follows sqrt
    assert sum(t.metadata.get("phase")=="cpu_dispatch" for t in tasks)==1


@pytest.mark.parametrize("enabled,use_ref", [(None,None),(True,None),(None,False)])
def test_unknown_cpu_controls_are_conditional_not_claimed_unfused(enabled,use_ref):
    tasks,_=tail(scenario(cpu_fusion=enabled,cpu_use_ref=use_ref))
    ops=mains(tasks);assert len(ops)==1
    audit=ops[0].metadata["final_norm_placement"]
    assert audit["fusion_status"]=="conditional_fused_lower_envelope"
    assert audit["unpriced_additional_dispatch_if_unfused"]==1
    assert audit["unpriced_additional_launch_if_unfused"]==0


@pytest.mark.parametrize("ngl,source", [(0,"cpu0"),(-1,"gpu0")])
def test_weight_binding_alone_does_not_prove_custom_graph_fusion(ngl,source):
    tasks,_=tail(scenario(ngl,source_bound=False),source=source)
    audit=mains(tasks)[0].metadata["final_norm_placement"]
    assert audit["graph_pattern_binding"]["status"]=="unknown_custom_graph"
    for key in ("adjacent","rms_single_use","rms_is_output","rms_is_view", "rms_is_mul_left_operand",
                "activation_contiguous_rows","weight_contiguous_rows","weight_contiguous_columns"):
        assert audit["fusion_conditions"][key] is None
    assert audit["fusion_status"]=="conditional_fused_lower_envelope"


def test_custom_graph_assertions_cannot_upgrade_source_binding():
    conditions={"same_split":True,"adjacent":True,"rms_single_use":True,
        "rms_is_output":False,"rms_is_view":False,"rms_is_mul_left_operand":True,
        "shape_connection_matches":True,"memory_ranges_compatible":True,"input_nb0":4,
        "activation_contiguous_rows":True,"weight_contiguous_rows":True}
    tasks,_=tail(scenario(-1,source_bound=False,conditions=conditions),source="gpu0")
    assert mains(tasks)[0].metadata["final_norm_placement"]["fusion_status"]=="conditional_fused_lower_envelope"


def test_cpu_fusion_allows_broadcast_weight_on_either_operand():
    tasks,_=tail(scenario(conditions={"rms_is_mul_left_operand":False}),rows=2)
    assert len(mains(tasks))==1


@pytest.mark.parametrize("ngl", [0,-1])
@pytest.mark.parametrize("norm_dtype", ["F16","F32"])
def test_plain_gguf_runtime_entry_without_f32_evidence_retains_analysis_fallback(ngl,norm_dtype):
    model=gguf_fixture_model(norm_dtype)
    base=build_reference_scenario()
    parallel=replace(base.placement.parallel,layer_to_stage={},
        rank_mapping=tuple(replace(rank,cim_component_id=None) for rank in base.placement.parallel.rank_mapping))
    authored=replace(base,model=model,
        placement=replace(base.placement,model_name=model.name,parallel=parallel),
        workload=WorkloadSpec("ordinary-GGUF",requests=(RequestSpec("r",0,4,2),)))
    cfg=LlamaCppRuntimeConfig(gpu_layers=ngl,op_offload=False,context=256,batch=64,ubatch=64)
    candidate=apply_llama_runtime_config(authored,cfg)
    assert not planner._f32_hidden_storage_enabled(candidate)
    # Compare to the previous metadata-only input surface; no native answer or
    # hardware/model file is read and the original model capacity is unchanged.
    def without_new_binding(value):
        if isinstance(value,dict):
            return {k:without_new_binding(v) for k,v in value.items() if k!="gguf_output_norm_binding"}
        return value
    legacy_model=replace(model,metadata=without_new_binding(dict(model.metadata)),
        graph=replace(model.graph,attributes=without_new_binding(dict(model.graph.attributes))))
    legacy=apply_llama_runtime_config(replace(authored,model=legacy_model),cfg)
    new=planner.compile_scenario(candidate);old=planner.compile_scenario(legacy)
    norms=[t for t in new.tasks if t.metadata.get("model_operator_id")=="final_norm"]
    assert norms and not mains(new.tasks)
    for task in norms:
        audit=task.metadata["final_norm_placement_fallback"]
        assert audit["status"]=="analytical_fallback" and audit["timing_completeness"]=="partial"
        assert audit["reason"]=="f32_hidden_storage_not_declared"
        assert audit["activation_bits"]==16 and audit["native_mechanism_applied"] is False
    signature=lambda schedule:[(t.name,t.dependencies,t.demands) for t in schedule.tasks]
    assert signature(new)==signature(old)
