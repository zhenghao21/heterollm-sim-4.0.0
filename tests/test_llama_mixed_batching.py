"""llama.cpp mixed prefill/decode adapter tests with the production runtime."""
from dataclasses import replace

import pytest

from heterollm_sim.control_plane_state import mapping_fingerprint_status
from heterollm_sim.ir import LayerSpec, LinearAttentionSpec, MTPPolicy, RequestSpec, SchedulerSpec, WorkloadSpec
from heterollm_sim.llama_scenario import apply_llama_runtime_config
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.reporting import run_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
from tests.model_helpers import model_from_layer_specs, replace_model_layer_specs


def authored(output_tokens=8):
    base = build_reference_scenario()
    layer = LayerSpec("dense0", "dense", hidden_size=32, intermediate_size=64,
        attention_heads=4, kv_heads=2, weight_bytes=65536)
    model = model_from_layer_specs("mixed-test", (layer,), vocabulary_size=64,
        max_sequence_length=512, embedding_weight_bytes=4096)
    workload = WorkloadSpec("mixed-test", requests=(
        RequestSpec("a", 0.0, 64, output_tokens), RequestSpec("b", 0.0, 192, 2)),
        scheduler=SchedulerSpec(mode="continuous", max_num_seqs=2,
            max_num_batched_tokens=64, max_num_ubatch_tokens=64,
            prefill_chunk_tokens=64, policy="decode_first", preemption_enabled=False))
    parallel = replace(base.placement.parallel, layer_to_stage={"dense0": 0},
        rank_mapping=tuple(replace(rank, cim_component_id=None) for rank in base.placement.parallel.rank_mapping))
    return replace(base, model=model, workload=workload,
        placement=replace(base.placement, model_name=model.name, parallel=parallel))


def lower(case=None, *, continuous=True, ubatch=64):
    return apply_llama_runtime_config(authored() if case is None else case,
        LlamaCppRuntimeConfig(batch=64, ubatch=ubatch, context=512, parallel=2,
            cont_batching=continuous))


def test_production_run_mixes_one_decode_with_63_prompt_rows():
    case = lower()
    fingerprint = mapping_fingerprint_status(case)
    assert fingerprint["fingerprint_present"] and not fingerprint["mapping_stale"]
    result = run_scenario(case)
    batches = result.serving.batches
    mixed = next(batch for batch in batches if batch.kind == "mixed")
    assert [(item.request_id, item.phase, item.token_count) for item in mixed.items] == [
        ("a", "decode", 1), ("b", "prefill", 63)]
    assert mixed.token_count == 64
    assert mixed.cost.metadata["execution_stage_source"] == "executed_task_dag_kernel_timeline"
    groups = mixed.cost.metadata["operator_invocation_groups"]
    assert [group["physical_ubatch_rows"] for group in groups] == [64]
    assert all(group["batching_semantics"] == "explicit_mixed_phase_physical_batch" for group in groups)
    assert sum(len(group["lanes"]) for group in groups) == mixed.token_count
    assert [item.logit_tokens for item in mixed.items] == [1, 0]
    # Every logical prompt row is appended once; only prompt completion and
    # generated rows request logits. The final mixed prompt batch has 1+3 rows.
    assert sum(item.token_count for batch in batches for item in batch.items
               if item.request_id == "b" and item.phase == "prefill") == 192
    tails = [batch for batch in batches if any(item.request_id == "b" and item.phase == "prefill" and item.logit_tokens == 1 for item in batch.items)]
    assert len(tails) == 1 and tails[0].token_count == 4
    assert result.serving.request_metrics["b"].first_token_ns < result.serving.request_metrics["a"].finish_ns
    assert result.manifest.metadata["control_plane"]["placement_fingerprint"] == fingerprint["input_fingerprint"]


def test_extending_first_output_does_not_delay_later_prompt_until_first_finishes():
    short = run_scenario(lower(authored(8))).serving
    long = run_scenario(lower(authored(32))).serving
    assert long.request_metrics["a"].finish_ns > short.request_metrics["a"].finish_ns
    assert long.request_metrics["b"].first_token_ns == pytest.approx(short.request_metrics["b"].first_token_ns)
    assert long.request_metrics["b"].first_token_ns < short.request_metrics["a"].finish_ns
    assert len([b for b in long.batches if b.kind == "mixed"]) == 4


def test_mixed_logical_batch_splits_to_physical_ubatch_without_losing_rows():
    result = run_scenario(lower(ubatch=16))
    mixed = next(batch for batch in result.serving.batches if batch.kind == "mixed")
    groups = mixed.cost.metadata["operator_invocation_groups"]
    assert mixed.token_count == 64
    assert [group["physical_ubatch_rows"] for group in groups] == [16] * 4
    lanes = [lane for group in groups for lane in group["lanes"]]
    assert len(lanes) == 64
    assert sum(lane["request_id"] == "a" for lane in lanes) == 1
    assert sum(lane["request_id"] == "b" for lane in lanes) == 63
    assert len({(lane["request_id"], lane["position"]) for lane in lanes}) == 64


def test_disabled_cont_batching_clears_mixed_setting():
    case = authored()
    case = replace(case, workload=replace(case.workload,
        scheduler=replace(case.workload.scheduler, mixed_phase_batching=True)))
    result = lower(case, continuous=False)
    assert result.workload.scheduler.mixed_phase_batching is False
    assert result.workload.metadata["llama_cpp_mixed_phase_batching"]["status"] == "disabled"


@pytest.mark.parametrize("kind,reason", [
    ("linear", "recurrent_physical_ubatch_unproven"),
    ("moe", "expert_batching_unproven"),
    ("mtp", "mtp_batching_unproven"),
    ("image", "non_text_request_batching_unproven"),
    ("adapter", "request_task_or_adapter_compatibility_unproven"),
])
def test_unsupported_graphs_never_inherit_mixed_batching(kind, reason):
    case = authored()
    if kind in {"linear", "moe"}:
        layer = LayerSpec("dense0", "dense", hidden_size=32, intermediate_size=64,
            attention_heads=4, kv_heads=2, weight_bytes=65536)
        if kind == "linear":
            layer = replace(layer, sequence_mixer="linear_attention", linear_attention=LinearAttentionSpec(
                key_heads=2, value_heads=2, key_head_dim=16, value_head_dim=16, conv_kernel_size=4))
        else:
            layer = replace(layer, kind="moe", num_experts=4, experts_per_token=2)
        case = replace(case, model=replace_model_layer_specs(case.model, (layer,)))
    elif kind == "mtp":
        case = replace(case, workload=replace(case.workload, mtp=MTPPolicy()))
    else:
        metadata = {"modality": "image"} if kind == "image" else {"lora": "adapter-a"}
        case = replace(case, workload=replace(case.workload,
            requests=(replace(case.workload.requests[0], metadata=metadata), case.workload.requests[1])))
    result = apply_llama_runtime_config(case,
        LlamaCppRuntimeConfig(batch=64, ubatch=64, context=512, parallel=2), materialize_placement=False)
    audit = result.workload.metadata["llama_cpp_mixed_phase_batching"]
    assert not result.workload.scheduler.mixed_phase_batching
    assert audit["status"] == "unsupported" and audit["reason"] == reason
    assert audit["accuracy_validated"] is False


def test_mixed_batching_execution_contract_never_claims_accuracy_validation():
    audit = lower().workload.metadata["llama_cpp_mixed_phase_batching"]
    assert audit["status"] == "enabled"
    assert audit["evidence_kind"] == "source_derived_execution_semantics"
    assert audit["accuracy_validated"] is False


from pathlib import Path
from heterollm_sim.runtime_adapters import derive_llama_hybrid_batch_contract


@pytest.fixture
def hybrid_contract(tmp_path):
    files = {
        'server.cpp': '''bool can_batch_with(server_slot & other_slot) const {
            return task->type == other_slot.task->type && are_lora_equal(lora, other_slot.lora);
        }
        bool can_split() const { return !task->need_embd(); }
        if (params_base.cont_batching || batch.size() == 0) {}
        while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {}''',
        'src/llama-memory-hybrid.cpp': '''const bool unified = (mem_attn->get_n_stream() == 1);
            ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            mem_recr->prepare(ubatches); mem_attn->prepare(ubatches);''',
        'src/llama-batch.cpp': '''llama_ubatch llama_batch_allocr::split_equal(uint32_t n_ubatch) {
            cur_idx[s] >= (int32_t) seq_set_map[cur_seq_set[s]].size();
            (idxs_per_seq[0].size() + 1)*n_seqs > n_ubatch;
            idxs.insert(idxs.end(), idxs_per_seq[s].begin(), idxs_per_seq[s].end());
        }''',
        'src/models/qwen35.cpp': '''GGML_ASSERT(ubatch.equal_seqs());
            GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);''',
        'captured.log': 'llama_context: n_seq_max = 4\nllama_context: n_ubatch = 64\nllama_context: n_rs_seq = 0\nllama_context: kv_unified = true\n',
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    return derive_llama_hybrid_batch_contract(tmp_path, server_source=tmp_path/'server.cpp',
                                            runtime_log=tmp_path/'captured.log')


def hybrid_authored(output_tokens=8):
    case = authored(output_tokens)
    recurrent = LayerSpec('dense0', 'dense', hidden_size=32, intermediate_size=64,
        attention_heads=4, kv_heads=2, weight_bytes=65536,
        sequence_mixer='linear_attention', linear_attention=LinearAttentionSpec(
            key_heads=2, value_heads=2, key_head_dim=16, value_head_dim=16, conv_kernel_size=4))
    model = model_from_layer_specs('hybrid-test', (recurrent,), vocabulary_size=64,
        max_sequence_length=512, embedding_weight_bytes=4096, architecture='qwen3_5_hybrid_transformer')
    return replace(case, model=model, placement=replace(case.placement, model_name=model.name))


def lower_hybrid(contract, *, output=8, ubatch=64, **config):
    return apply_llama_runtime_config(hybrid_authored(output),
        LlamaCppRuntimeConfig(batch=64, ubatch=ubatch, context=512, parallel=2, **config),
        recurrent_batching_contract=contract)


def test_hybrid_source_proof_captures_geometry_not_latency(hybrid_contract):
    proof = hybrid_contract
    assert proof['status'] == 'source_derived'
    assert proof['recurrent_rollback_snapshots'] == 0 and proof['kv_unified'] is True
    assert proof['captured_sequence_capacity'] == 4
    assert len(proof['source_sha256']) == 4
    assert proof['accuracy_validated'] is False and proof['native_latency_used'] is False


def test_source_bound_hybrid_mixes_decode_and_prompt_with_rectangular_microbatches(hybrid_contract):
    case = lower_hybrid(hybrid_contract)
    assert case.workload.scheduler.mixed_phase_batching is True
    assert case.workload.metadata['llama_cpp_mixed_phase_batching']['reason'] == 'source_bound_hybrid_equal_length_ubatches'
    assert case.workload.metadata['supports_equal_length_stateful_ubatches'] is True
    result = run_scenario(case)
    mixed = next(batch for batch in result.serving.batches if batch.kind == 'mixed')
    assert [(item.request_id, item.phase, item.token_count) for item in mixed.items] == [('a', 'decode', 1), ('b', 'prefill', 63)]
    groups = mixed.cost.metadata['operator_invocation_groups']
    assert [group['physical_ubatch_rows'] for group in groups] == [2, 62]
    assert all(group['batching_semantics'] == 'explicit_equal_length_stateful_ubatch' for group in groups)
    for group in groups:
        counts = {}
        for lane in group['lanes']:
            counts[lane['request_id']] = counts.get(lane['request_id'], 0) + 1
        assert len(set(counts.values())) == 1
    assert sum(item.token_count for batch in result.serving.batches for item in batch.items
               if item.request_id == 'b' and item.phase == 'prefill') == 192
    assert result.serving.request_metrics['b'].first_token_ns < result.serving.request_metrics['a'].finish_ns


def test_hybrid_rectangular_groups_respect_smaller_physical_ubatch(hybrid_contract):
    result = run_scenario(lower_hybrid(hybrid_contract, ubatch=16))
    mixed = next(batch for batch in result.serving.batches if batch.kind == 'mixed')
    groups = mixed.cost.metadata['operator_invocation_groups']
    assert [group['physical_ubatch_rows'] for group in groups] == [2, 16, 16, 16, 14]
    lanes = [lane for group in groups for lane in group['lanes']]
    assert len(lanes) == 64
    assert len({(lane['request_id'], lane['position']) for lane in lanes}) == 64


def test_hybrid_later_prompt_no_longer_waits_for_all_earlier_output_tokens(hybrid_contract):
    short = run_scenario(lower_hybrid(hybrid_contract, output=8)).serving
    long = run_scenario(lower_hybrid(hybrid_contract, output=32)).serving
    assert long.request_metrics['a'].finish_ns > short.request_metrics['a'].finish_ns
    assert long.request_metrics['b'].first_token_ns == pytest.approx(short.request_metrics['b'].first_token_ns)
    assert long.request_metrics['b'].first_token_ns < short.request_metrics['a'].finish_ns


@pytest.mark.parametrize('field,value', [('recurrent_rollback_snapshots', 1), ('kv_unified', False),
    ('captured_sequence_capacity', 1), ('captured_ubatch_capacity', 16), ('native_latency_used', True),
    ('status', 'unverified'), ('source_sha256', {}), ('architectures', ['other'])])
def test_hybrid_invalid_or_out_of_scope_proof_stays_closed(hybrid_contract, field, value):
    proof = {**hybrid_contract, field: value}
    case = apply_llama_runtime_config(hybrid_authored(),
        LlamaCppRuntimeConfig(batch=64, ubatch=64, context=512, parallel=2),
        materialize_placement=False, recurrent_batching_contract=proof)
    assert not case.workload.scheduler.mixed_phase_batching
    assert case.workload.metadata['llama_cpp_mixed_phase_batching']['reason'] == 'recurrent_physical_ubatch_unproven'


def test_hybrid_nonunified_kv_cannot_reuse_rectangular_unified_proof(hybrid_contract):
    case = lower_hybrid(hybrid_contract, kv_unified=False)
    assert not case.workload.scheduler.mixed_phase_batching


def test_hybrid_proof_does_not_enable_adapters_or_mtp(hybrid_contract):
    for case in (replace(hybrid_authored(), workload=replace(hybrid_authored().workload, mtp=MTPPolicy())),
                 replace(hybrid_authored(), workload=replace(hybrid_authored().workload,
                     requests=(replace(hybrid_authored().workload.requests[0], metadata={'lora': 'adapter'}),)) )):
        lowered = apply_llama_runtime_config(case,
            LlamaCppRuntimeConfig(batch=64, ubatch=64, context=512, parallel=2),
            materialize_placement=False, recurrent_batching_contract=hybrid_contract)
        assert not lowered.workload.scheduler.mixed_phase_batching


def test_changed_source_or_rollback_runtime_log_cannot_derive_proof(tmp_path, hybrid_contract):
    root = Path(next(iter(hybrid_contract['source_sha256']))).parent
    server = root/'server.cpp'
    runtime_log = root/'captured.log'
    runtime_log.write_text('n_seq_max = 4\nn_ubatch = 64\nn_rs_seq = 2\nkv_unified = true\n', encoding='utf-8')
    with pytest.raises(ValueError, match='zero recurrent rollback'):
        derive_llama_hybrid_batch_contract(root, server_source=server, runtime_log=runtime_log)
    runtime_log.write_text('n_seq_max = 4\nn_ubatch = 64\nn_rs_seq = 0\nkv_unified = true\n', encoding='utf-8')
    (root/'src/llama-memory-hybrid.cpp').write_text('unrecognized backend', encoding='utf-8')
    with pytest.raises(ValueError, match='unrecognized hybrid'):
        derive_llama_hybrid_batch_contract(root, server_source=server, runtime_log=runtime_log)



def test_relowering_invalidates_owned_capabilities_when_kv_mode_changes(hybrid_contract):
    original = lower_hybrid(hybrid_contract)
    lowered = apply_llama_runtime_config(original,
        LlamaCppRuntimeConfig(batch=64, ubatch=64, context=512, parallel=2, kv_unified=False),
        materialize_placement=False)
    assert not lowered.workload.scheduler.mixed_phase_batching
    assert lowered.workload.metadata['supports_batched_stateful_execution'] is False
    assert lowered.workload.metadata['supports_equal_length_stateful_ubatches'] is False


def test_bound_hybrid_final_mapping_fingerprint_is_current(hybrid_contract):
    status = mapping_fingerprint_status(lower_hybrid(hybrid_contract))
    assert status['fingerprint_present'] and not status['mapping_stale']
