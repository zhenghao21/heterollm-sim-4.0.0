"""GET_ROWS storage/traffic regression; no native or model-file execution."""
from dataclasses import replace

import pytest

from heterollm_sim import planner
from heterollm_sim.ir import LayerSpec, RequestSpec, WorkloadSpec, SchedulerSpec, MTPPolicy
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.llama_tensor_storage import (
    SCHEMA, SOURCE_KEY, AUDIT_KEY, apply_llama_tensor_storage_contract,
    qualify_llama_tensor_storage_contract, resolve_embedding_gather_access,
    derive_llama_tensor_storage_contract,
)
from tests.model_helpers import model_from_layer_specs


@pytest.fixture
def contract():
    return {'schema':SCHEMA,'status':'source_derived','source_sha256':{'fixture.cpp':'a'*64},
        'embedding_index_bits':32,'embedding_output_storage_bits':32,
        'get_rows_access':'selected_packed_rows_per_index',
        'supported_weight_types':{'cpu':['Q8_0','F16','F32'],'gpu':['Q8_0','F16','F32']},
        'architectures':['qwen2','qwen2_decoder','llama','llama_decoder','qwen3_5_hybrid_transformer'],
        'hidden_storage_evidence':{'ggml_add_impl':True,'ggml_rms_norm_impl':True,'ggml_unary_impl':True,'ggml_ssm_conv':True,'ggml_gated_delta_net':True},
        'scope':'ordinary typed text GGUF test fixture','accuracy_validated':False,'native_latency_used':False}


def scenario(*, hidden=32, vocab=10000, row_bytes=34, target='cpu0', backing='hostmem0', weight_type='Q8_0'):
    base=build_reference_scenario()
    layer=LayerSpec('dense0','dense',hidden_size=hidden,intermediate_size=2*hidden,
                    attention_heads=4,kv_heads=2,weight_bytes=4096)
    total=vocab*row_bytes
    model=model_from_layer_specs('arbitrary-name-not-a-routing-key',(layer,),vocabulary_size=vocab,
        max_sequence_length=256,embedding_weight_bytes=total,architecture='qwen2',
        metadata={'gguf_sha256':'b'*64,'gguf_embedding_binding':{'name':'token_embd.weight',
            'shape':[hidden,vocab],'type':weight_type,'n_bytes':total,'offset':0}})
    parallel=replace(base.placement.parallel,layer_to_stage={'dense0':0},
        rank_mapping=tuple(replace(r,cim_component_id=None) for r in base.placement.parallel.rank_mapping))
    placement=replace(base.placement,model_name=model.name,parallel=parallel,
        op_to_component={'embedding':target},tensor_to_component={'embedding_weights':backing},
        tensor_bytes={'embedding_weights':total},metadata={})
    workload=WorkloadSpec('fixture',requests=(RequestSpec('r',0.,64,1),),
        scheduler=SchedulerSpec(max_num_seqs=1,preemption_enabled=False))
    return replace(base,model=model,placement=placement,workload=workload)


def embedding_tasks(case, rows=64):
    builder=planner._TaskBuilder(case.workload.requests[0])
    planner._compile_parallel_embedding(builder,case,planner._parallel_plan(case),planner._topology_router(case),
                                        'fixture',(),token_batch=rows)
    return builder.tasks


def compute_task(tasks):
    return next(t for t in tasks if t.metadata.get('event_kind')=='embedding' and t.category.value=='memory')


def test_without_contract_preserves_legacy_whole_table_cpu_traffic():
    case=scenario();task=compute_task(embedding_tasks(case));total=case.model.embedding_weight_bytes
    assert task.metadata['weight_read_bytes']==total
    assert task.metadata['output_bytes']==64*32*2
    assert task.metadata['native_get_rows_storage']['requested'] is False
    memory=next(d for d in task.demands if d.resource_id=='cpu0.memory')
    assert memory.bytes_moved==total+64*32*2


def test_local_cpu_charges_only_selected_packed_rows_ids_and_f32_writes(contract):
    case=apply_llama_tensor_storage_contract(scenario(),contract);task=compute_task(embedding_tasks(case))
    assert task.metadata['rank_weight_capacity_bytes']==340000
    assert task.metadata['weight_read_bytes']==64*34
    assert task.metadata['lookup_index_read_bytes']==64*4
    assert task.metadata['output_bytes']==64*32*4
    memory=next(d for d in task.demands if d.resource_id=='cpu0.memory')
    assert memory.bytes_moved==64*34+64*4+64*32*4
    assert task.metadata['embedding_dequant_compute']=='unmodeled_selected_rows_only'
    assert task.metadata['embedding_dequant_selected_elements']==64*32
    assert task.metadata['native_get_rows_storage']['dequantization_compute']['throughput_assigned'] is False


def test_actual_27b_embedding_geometry_separates_1_35gb_capacity_from_row_traffic(contract):
    case=apply_llama_tensor_storage_contract(scenario(hidden=5120,vocab=248320,row_bytes=5440),contract)
    task=compute_task(embedding_tasks(case));m=task.metadata
    assert m['rank_weight_capacity_bytes']==1350860800
    assert m['lookup_read_bytes']==348160
    assert m['lookup_index_read_bytes']==256
    assert m['lookup_write_bytes']==1310720
    assert next(d.bytes_moved for d in task.demands if d.resource_id=='cpu0.memory')==1659136


def test_gpu_local_source_uses_same_geometry_without_model_name_gate(contract):
    case=apply_llama_tensor_storage_contract(scenario(target='gpu0',backing='hbm0'),contract)
    task=compute_task(embedding_tasks(case))
    assert task.metadata['native_get_rows_storage']['qualified'] is True
    assert task.metadata['weight_read_bytes']==64*34
    assert task.metadata['lookup_index_read_bytes']==256
    assert task.metadata['output_bytes']==8192


def test_remote_table_staging_keeps_entire_capacity_before_local_gather(contract):
    case=apply_llama_tensor_storage_contract(scenario(target='cpu0',backing='hbm0'),contract)
    tasks=embedding_tasks(case);task=compute_task(tasks)
    access=[t for t in tasks if t.metadata.get('event_kind')=='model_weight_access']
    assert access
    assert access[0].metadata['weight_read_bytes']==340000
    assert access[0].metadata['weight_access_semantics']=='full_weight_staging_before_row_lookup'
    assert any(t.metadata.get('weight_read_bytes')==340000 and t.metadata.get('weight_access_semantics')=='full_weight_staging_before_row_lookup' for t in tasks)
    transfers = [t for t in tasks if t.metadata.get('event_kind') == 'model_weight_read']
    assert transfers and all(t.metadata['bytes'] == 340000 for t in transfers)
    assert all(d.bytes_moved == 340000 for t in transfers for d in t.demands)
    assert task.metadata['weight_read_bytes']==2176
    assert task.metadata['lookup_write_bytes']==8192


def test_embedding_only_ablation_does_not_force_other_hidden_storage(contract):
    case=apply_llama_tensor_storage_contract(scenario(),contract,f32_hidden_storage=False)
    assert case.workload.metadata.get('llama_cpp_f32_hidden_storage',False) is False
    task=compute_task(embedding_tasks(case))
    assert task.metadata['output_bytes']==8192


@pytest.mark.parametrize('change,reason',[
    ('missing_identity','typed_gguf_identity_required'),('bad_shape','contiguous_2d_embedding_shape_required'),
    ('view','embedding_view_or_strides_unproven'),('mtp','mtp_hidden_storage_unproven'),
    ('adapter','dynamic_embedding_or_adapter_path_unproven'),('nontext','dynamic_embedding_or_adapter_path_unproven')])
def test_source_scope_is_not_applied_to_dynamic_or_unbound_tensors(contract,change,reason):
    case=scenario();metadata=dict(case.model.metadata)
    if change=='missing_identity':
        metadata.pop('gguf_sha256')
        attrs=dict(case.model.graph.attributes)
        attrs.pop('gguf_sha256', None)
        if isinstance(attrs.get('metadata'), dict):
            attrs['metadata']={k:v for k,v in attrs['metadata'].items() if k!='gguf_sha256'}
        case=replace(case,model=replace(case.model,graph=replace(case.model.graph,attributes=attrs)))
    elif change=='bad_shape':metadata['gguf_embedding_binding']={**metadata['gguf_embedding_binding'],'shape':[32,10000,2]}
    elif change=='view':metadata['gguf_embedding_binding']={**metadata['gguf_embedding_binding'],'is_view':True}
    elif change=='mtp':case=replace(case,workload=replace(case.workload,mtp=MTPPolicy()))
    elif change=='adapter':case=replace(case,workload=replace(case.workload,metadata={'lora':'adapter-a'}))
    elif change=='nontext':case=replace(case,workload=replace(case.workload,metadata={'modality':'image'}))
    case=replace(case,model=replace(case.model,metadata=metadata))
    case=apply_llama_tensor_storage_contract(case,contract)
    assert not case.workload.metadata[AUDIT_KEY]['qualified']
    assert reason in case.workload.metadata[AUDIT_KEY]['reasons']
    assert case.workload.metadata.get('llama_cpp_f32_hidden_storage',False) is False


def test_layout_or_capacity_mismatch_falls_back_explicitly(contract):
    case=apply_llama_tensor_storage_contract(scenario(),contract)
    result=resolve_embedding_gather_access(case,token_rows=64,hidden_width=33,
        rank_weight_capacity_bytes=340001,target_kind='cpu',tp_degree=1)
    assert not result['qualified']
    assert 'physical_embedding_shape_or_shard_mismatch' in result['reasons']
    assert 'physical_packed_embedding_row_bytes_unverified' in result['reasons']


def test_unknown_target_or_quant_type_is_not_claimed(contract):
    case=apply_llama_tensor_storage_contract(scenario(weight_type='UNSUPPORTED'),contract)
    result=resolve_embedding_gather_access(case,token_rows=64,hidden_width=32,
        rank_weight_capacity_bytes=340000,target_kind='cpu',tp_degree=1)
    assert not result['qualified'] and 'get_rows_target_or_weight_type_unproven' in result['reasons']


def test_contract_never_uses_a_latency_fitted_speed(contract):
    case=apply_llama_tensor_storage_contract(scenario(),{**contract,'native_latency_used':True})
    assert not case.workload.metadata[AUDIT_KEY]['qualified']
    assert 'tensor_storage_source_contract_unverified' in case.workload.metadata[AUDIT_KEY]['reasons']

@pytest.mark.parametrize('change', ['hash', 'hidden_rule'])
def test_unverified_source_evidence_cannot_enable_hidden_storage(contract, change):
    if change == 'hash':
        contract = {**contract, 'source_sha256': {'fixture.cpp': 'invalid'}}
    else:
        contract = {**contract, 'hidden_storage_evidence': {
            **contract['hidden_storage_evidence'], 'ggml_ssm_conv': False}}
    case = apply_llama_tensor_storage_contract(scenario(), contract)
    assert not qualify_llama_tensor_storage_contract(case)['qualified']
    assert case.workload.metadata.get('llama_cpp_f32_hidden_storage', False) is False


@pytest.mark.parametrize('binding_change', [
    {'shape': [64, 10000]}, {'n_bytes': 340001}, {'type': 'I32'}, {'type': 'UNSUPPORTED'},
])
def test_unbound_storage_cannot_enable_whole_graph_f32(contract, binding_change):
    case = scenario()
    metadata = {**case.model.metadata, 'gguf_embedding_binding': {
        **case.model.metadata['gguf_embedding_binding'], **binding_change}}
    case = replace(case, model=replace(case.model, metadata=metadata))
    case = apply_llama_tensor_storage_contract(case, contract)
    assert not case.workload.metadata[AUDIT_KEY]['qualified']
    assert case.workload.metadata.get('llama_cpp_f32_hidden_storage', False) is False


@pytest.mark.parametrize('original', [False, True])
def test_embedding_only_reapplication_restores_existing_f32_preference(contract, original):
    case = scenario()
    case = replace(case, workload=replace(case.workload, metadata={
        'llama_cpp_f32_hidden_storage': original, 'unrelated': 'preserved'}))
    full = apply_llama_tensor_storage_contract(case, contract)
    assert full.workload.metadata['llama_cpp_f32_hidden_storage'] is True
    embedding_only = apply_llama_tensor_storage_contract(full, contract, f32_hidden_storage=False)
    assert embedding_only.workload.metadata['llama_cpp_f32_hidden_storage'] is original
    assert embedding_only.workload.metadata['unrelated'] == 'preserved'
    assert embedding_only.model == case.model
    assert embedding_only.placement == case.placement
    assert compute_task(embedding_tasks(embedding_only)).metadata['output_bytes'] == 8192


def test_f32_rows_are_copied_without_claiming_dequantization_work(contract):
    case = apply_llama_tensor_storage_contract(scenario(weight_type='F32', row_bytes=128), contract)
    task = compute_task(embedding_tasks(case))
    assert task.metadata['weight_read_bytes'] == 8192
    assert task.metadata['output_bytes'] == 8192
    assert task.metadata['embedding_dequant_compute'] == 'not_required'
    assert task.metadata['native_get_rows_storage']['dequantization_compute']['throughput_assigned'] is False


def write_source_fixture(root):
    operators = """
struct ggml_tensor * ggml_get_rows(void) {
    assert(b->type == GGML_TYPE_I32);
    enum ggml_type type = GGML_TYPE_F32;
    if (a->type == GGML_TYPE_I32) { type = GGML_TYPE_I32; }
    result->op = GGML_OP_GET_ROWS;
}
struct ggml_tensor * ggml_mul_mat(void) {
    return ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
}
"""
    for name in ('ggml_add_impl', 'ggml_rms_norm_impl', 'ggml_unary_impl'):
        operators += ('static struct ggml_tensor * ' + name + '(void) {'
                      'return inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);}')
    for name in ('ggml_ssm_conv', 'ggml_gated_delta_net'):
        operators += ('struct ggml_tensor * ' + name + '(void) {'
                      'return ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);}')
    sources = {
        'ggml/src/ggml.c': operators,
        'ggml/src/ggml-cpu/ops.cpp': """
static void ggml_compute_forward_get_rows_q(void) {
    const int64_t nr = ggml_nelements(src1);
    auto dequantize_row_q = ggml_get_type_traits(type)->to_float;
    int64_t i01 = *(int32_t *) src1->data;
    dequantize_row_q(src0->data + i01*nb01, (float *) dst->data, nc);
}
void ggml_compute_forward_get_rows(void) {
    switch (src0->type) {
        case GGML_TYPE_Q8_0: case GGML_TYPE_F32: case GGML_TYPE_I32: break;
    }
}
""",
        'ggml/src/ggml-cuda/getrows.cu': """
void k_get_rows(void) {
    const int i01 = src1[id];
    const auto src0_row = src0 + i01*nb01;
    dst_row[id] = dequantize_kernel(src0_row, id);
    switch (src0->type) { case GGML_TYPE_Q8_0: case GGML_TYPE_F32: break; }
}
""",
        'src/llama-graph.cpp': """
ggml_tensor * llm_graph_context::build_inp_embd(void) {
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
    cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    for (const auto & lora : *loras) { apply(lora); }
}
""",
    }
    for relative, text in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')


def test_source_contract_derives_rules_and_hashes_without_executing_native(tmp_path):
    write_source_fixture(tmp_path)
    derived = derive_llama_tensor_storage_contract(tmp_path)
    assert derived['schema'] == SCHEMA
    assert len(derived['source_sha256']) == 4
    assert all(len(value) == 64 for value in derived['source_sha256'].values())
    assert derived['supported_weight_types'] == {'cpu': ['F32', 'Q8_0'], 'gpu': ['F32', 'Q8_0']}
    assert derived['native_latency_used'] is False
    assert derived['accuracy_validated'] is False
    assert all(derived['hidden_storage_evidence'].values())


def test_source_contract_rejects_missing_indexed_row_rule(tmp_path):
    write_source_fixture(tmp_path)
    path = tmp_path / 'ggml/src/ggml-cpu/ops.cpp'
    path.write_text(path.read_text(encoding='utf-8').replace('src0->data + i01*nb01', 'src0->data'), encoding='utf-8')
    with pytest.raises(ValueError, match='cpu_indexed_quantized_rows'):
        derive_llama_tensor_storage_contract(tmp_path)


def test_native_graph_builder_nested_metadata_is_qualified(contract):
    case = scenario()
    identity = {key: case.model.metadata[key] for key in ("gguf_sha256", "gguf_embedding_binding")}
    model = replace(case.model,
        graph=replace(case.model.graph, attributes={**{k:v for k,v in case.model.graph.attributes.items() if k not in identity}, "metadata": identity}),
        metadata={"metadata": identity})
    applied = apply_llama_tensor_storage_contract(replace(case, model=model), contract)
    audit = qualify_llama_tensor_storage_contract(applied)
    assert audit["qualified"] is True
    task = compute_task(embedding_tasks(applied))
    assert task.metadata["lookup_read_bytes"] == 64 * 34


def test_conflicting_flat_and_nested_identity_is_not_silently_promoted(contract):
    case = scenario()
    identity = {key: case.model.metadata[key] for key in ("gguf_sha256", "gguf_embedding_binding")}
    conflicting = {**identity, "gguf_sha256": "c" * 64}
    model = replace(case.model, graph=replace(case.model.graph, attributes={**{k:v for k,v in case.model.graph.attributes.items() if k not in identity}, "metadata": identity}),
                    metadata={**conflicting, "metadata": identity})
    audit = qualify_llama_tensor_storage_contract(apply_llama_tensor_storage_contract(replace(case, model=model), contract))
    assert audit["qualified"] is False
    assert "conflicting_nested_gguf_evidence" in audit["reasons"]
