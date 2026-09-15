"""Source-qualified slot order for isolated fresh cohorts; no native execution."""
from dataclasses import replace
from pathlib import Path

import pytest

from heterollm_sim.control_plane_state import mapping_fingerprint_status
from heterollm_sim.ir import RequestSpec, MTPPolicy
from heterollm_sim.llama_scenario import apply_llama_runtime_config
from heterollm_sim.reporting import run_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig, derive_llama_slot_order_contract
from tests.test_llama_mixed_batching import authored, hybrid_authored, hybrid_contract


@pytest.fixture
def slot_contract(tmp_path):
    source = tmp_path / 'server-context.cpp'
    source.write_text('''
    void iterate(std::vector<server_slot> & slots, callback_t callback) {
        for (auto & slot : slots) { callback(slot); }
    }
    bool can_batch_with(server_slot & other_slot) const {
        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }
    void update_slots() {
        if (params_base.cont_batching || batch.size() == 0) {
            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) { return; }
                if (slot.state == SLOT_STATE_STARTED) {
                    slot.stats.update_prompt_start();
                    slot.state = SLOT_STATE_PROCESSING_PROMPT;
                }
                while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
                    batch.add(slot.id, cur_tok);
                }
            });
        }
    }
    ''', encoding='utf-8')
    return derive_llama_slot_order_contract(source)


def cohort(*, prompts=(192, 192), outputs=(8, 8), hybrid=False):
    case = hybrid_authored() if hybrid else authored()
    reqs = tuple(RequestSpec(chr(97+i), 0.0, p, o) for i, (p, o) in enumerate(zip(prompts, outputs)))
    return replace(case, workload=replace(case.workload, requests=reqs, request_count=len(reqs)))


def lower(case, contract=None, *, parallel=None, recurrent=None, **kwargs):
    return apply_llama_runtime_config(case,
        LlamaCppRuntimeConfig(batch=64, ubatch=64, context=512,
            parallel=parallel or len(case.workload.requests) or 2, **kwargs),
        slot_order_contract=contract, recurrent_batching_contract=recurrent)


def tokens_by_request(result):
    totals = {}
    for batch in result.serving.batches:
        assert sum(item.token_count for item in batch.items) == batch.token_count
        assert batch.token_count <= 64
        for item in batch.items:
            key = (item.request_id, item.phase)
            totals[key] = totals.get(key, 0) + item.token_count
    return totals


def test_no_slot_contract_preserves_existing_round_robin_default():
    case = lower(cohort())
    assert case.workload.scheduler.phase_candidate_order == 'least_recently_served'
    assert case.workload.metadata['llama_cpp_slot_order']['status'] == 'not_requested'
    result = run_scenario(case)
    assert [batch.request_ids for batch in result.serving.batches[:4]] == [('a',), ('b',), ('a',), ('b',)]


def test_slot_contract_keeps_current_prompt_until_batch_boundary(slot_contract):
    case = lower(cohort(), slot_contract)
    assert case.workload.scheduler.phase_candidate_order == 'stable_admission'
    audit = case.workload.metadata['llama_cpp_slot_order']
    assert audit['qualified'] and audit['preserves_engine_start_definition']
    result = run_scenario(case)
    assert [batch.request_ids for batch in result.serving.batches[:3]] == [('a',), ('a',), ('a',)]
    assert [batch.token_count for batch in result.serving.batches[:3]] == [64, 64, 64]
    first_mixed = next(batch for batch in result.serving.batches if batch.kind == 'mixed')
    assert [(item.request_id, item.phase, item.token_count) for item in first_mixed.items] == [('a', 'decode', 1), ('b', 'prefill', 63)]
    totals = tokens_by_request(result)
    assert totals == {('a', 'prefill'): 192, ('b', 'prefill'): 192, ('a', 'decode'): 7, ('b', 'decode'): 7}


def test_engine_clock_still_begins_at_first_selected_prompt_processing(slot_contract):
    result = run_scenario(lower(cohort(), slot_contract))
    starts = [event for event in result.serving.events if event.event_type == 'engine_request_begin']
    assert [event.request_id for event in starts] == ['a', 'b']
    assert len(starts) == 2
    for event in starts:
        assert event.details['boundary'] == 'first_prompt_batch_processing'
        assert event.timestamp_ns == result.serving.request_metrics[event.request_id].engine_start_ns
    assert result.serving.request_metrics['b'].engine_start_ns >= result.serving.request_metrics['a'].first_token_ns
    # This is a workload-order change. No timestamp is selected by comparing
    # against a native latency or by subtracting another request's service.
    assert result.serving.request_metrics['b'].first_token_ns > result.serving.request_metrics['b'].engine_start_ns


def test_four_fresh_slots_combine_stable_order_and_proven_hybrid_microbatches(slot_contract, hybrid_contract):
    case = cohort(prompts=(192,)*4, outputs=(8,)*4, hybrid=True)
    case = lower(case, slot_contract, recurrent=hybrid_contract)
    assert case.workload.scheduler.mixed_phase_batching
    result = run_scenario(case)
    totals = tokens_by_request(result)
    for rid in ('a', 'b', 'c', 'd'):
        assert totals[(rid, 'prefill')] == 192
        assert totals[(rid, 'decode')] == 7
    mixed = next(batch for batch in result.serving.batches if batch.kind == 'mixed')
    groups = mixed.cost.metadata['operator_invocation_groups']
    assert [g['physical_ubatch_rows'] for g in groups] == [2, 62]
    assert all(g['batching_semantics'] == 'explicit_equal_length_stateful_ubatch' for g in groups)
    assert [batch.request_ids for batch in result.serving.batches[:3]] == [('a',), ('a',), ('a',)]
    fingerprint = mapping_fingerprint_status(case)
    assert fingerprint['fingerprint_present'] and not fingerprint['mapping_stale']


def test_existing_prompt_stop_offsets_remain_intact(slot_contract):
    case = cohort(prompts=(192,192), outputs=(8,8))
    case = replace(case, workload=replace(case.workload,
        scheduler=replace(case.workload.scheduler, prefill_stop_offsets=(68,4))))
    lowered = lower(case, slot_contract)
    assert lowered.workload.scheduler.prefill_stop_offsets == (68,4)
    result = run_scenario(lowered)
    totals = tokens_by_request(result)
    assert totals[('a','prefill')] == totals[('b','prefill')] == 192
    assert all(batch.token_count<=64 for batch in result.serving.batches)


@pytest.mark.parametrize('kind,reason', [
    ('late','dynamic_or_staggered_arrivals_unproven'),
    ('more','request_count_exceeds_fresh_slots'),
    ('reuse','nonfresh_or_reused_slots_unproven'),
    ('nonfresh','nonfresh_or_reused_slots_unproven'),
    ('stream','arrival_stream_unproven'),
    ('priority','priority_or_deadline_order_unproven'),
    ('deadline','priority_or_deadline_order_unproven'),
    ('aging','priority_aging_unproven'),
    ('preempt','preemption_or_slot_reuse_unproven'),
    ('shortchunk','prefill_chunk_shorter_than_native_batch'),
    ('mtp','mtp_slot_reuse_unproven'),
    ('adapter','slot_task_compatibility_unproven'),
    ('implicit','explicit_closed_request_set_required'),
])
def test_out_of_scope_cohorts_do_not_inherit_slot_order(slot_contract, kind, reason):
    case = cohort();work = case.workload
    if kind == 'late': work = replace(work, requests=(work.requests[0], replace(work.requests[1], arrival_ns=1.0)))
    elif kind == 'more': work = replace(work, requests=(*work.requests, RequestSpec('c',0.0,192,8)))
    elif kind == 'reuse': work = replace(work, metadata={**work.metadata,'slot_reuse':True})
    elif kind == 'nonfresh': work = replace(work, metadata={**work.metadata,'initial_slots_empty':False})
    elif kind == 'stream': work = replace(work, arrival_rate_rps=1.0)
    elif kind == 'priority': work = replace(work, requests=(work.requests[0],replace(work.requests[1],priority=1)))
    elif kind == 'deadline': work = replace(work, requests=(work.requests[0],replace(work.requests[1],deadline_ns=1e9)))
    elif kind == 'aging': work = replace(work, scheduler=replace(work.scheduler,policy='decode_first_aging'))
    elif kind == 'preempt': work = replace(work, scheduler=replace(work.scheduler,preemption_enabled=True))
    elif kind == 'shortchunk': work = replace(work, scheduler=replace(work.scheduler,prefill_chunk_tokens=16))
    elif kind == 'mtp': work = replace(work,mtp=MTPPolicy())
    elif kind == 'adapter': work = replace(work,metadata={'lora':'adapter-a'})
    elif kind == 'implicit': work = replace(work,requests=(),request_count=2,prompt_tokens=192,output_tokens=8)
    case=replace(case,workload=work)
    lowered=lower(case,slot_contract,parallel=2)
    audit=lowered.workload.metadata['llama_cpp_slot_order']
    assert not audit['qualified'] and reason in audit['reasons']
    assert lowered.workload.scheduler.phase_candidate_order=='least_recently_served'


def test_disabling_previously_bound_order_restores_authored_policy(slot_contract):
    enabled=lower(cohort(),slot_contract)
    disabled=lower(enabled,{**slot_contract,'status':'disabled'})
    assert disabled.workload.scheduler.phase_candidate_order=='least_recently_served'
    assert not disabled.workload.metadata['llama_cpp_slot_order']['applied']


def test_dynamic_arrival_after_relowering_invalidates_old_qualification(slot_contract):
    case=lower(cohort(),slot_contract)
    work=replace(case.workload,requests=(case.workload.requests[0],replace(case.workload.requests[1],arrival_ns=1.0)))
    invalid=lower(replace(case,workload=work))
    assert invalid.workload.scheduler.phase_candidate_order=='least_recently_served'
    assert 'dynamic_or_staggered_arrivals_unproven' in invalid.workload.metadata['llama_cpp_slot_order']['reasons']


def test_native_source_order_or_chain_change_rejects_contract(slot_contract):
    path=Path(next(iter(slot_contract['source_sha256'])))
    with pytest.raises(ValueError,match='source chain'):
        derive_llama_slot_order_contract(path,source_chain={'source_sha256':{str(path):'0'*64}})
    path.write_text(path.read_text(encoding='utf-8').replace('callback(slot);','std::rotate(slots); callback(slot);'),encoding='utf-8')
    with pytest.raises(ValueError,match='fixed slot iteration'):
        derive_llama_slot_order_contract(path)


def test_slot_contract_cannot_be_marked_accuracy_validated(slot_contract):
    invalid=lower(cohort(),{**slot_contract,'accuracy_validated':True})
    assert invalid.workload.scheduler.phase_candidate_order=='least_recently_served'
    assert 'slot_order_source_contract_unverified' in invalid.workload.metadata['llama_cpp_slot_order']['reasons']


def test_slot_source_reader_ignores_comment_and_string_braces(slot_contract):
    path=Path(next(iter(slot_contract['source_sha256'])))
    source=path.read_text(encoding='utf-8')
    source=source.replace('void update_slots() {',
        'void update_slots() { // closing pseudo-code brace }\n'
        '/* another brace } */ const char *label = "}";\n'
        'const char *raw = R"tag( } // comment inside raw string )tag";\n')
    path.write_text(source,encoding='utf-8')
    proof=derive_llama_slot_order_contract(path)
    assert proof['status']=='source_derived'
    assert proof['preserves_engine_start_definition'] is True



def test_native_pre_decode_call_chain_is_followed(slot_contract):
    path=Path(next(iter(slot_contract['source_sha256'])))
    source=path.read_text(encoding='utf-8').replace('void update_slots() {',
        'void update_slots() { pre_decode(); }\n void pre_decode() {')
    path.write_text(source,encoding='utf-8')
    proof=derive_llama_slot_order_contract(path)
    assert 'server_context::pre_decode' in proof['source_symbols']
