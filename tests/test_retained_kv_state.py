"""Structural-only retained slot lifecycle tests; no native calls or timing fit."""
import copy
from dataclasses import replace

import pytest
from heterollm_sim import planner
from heterollm_sim.retained_kv_state import RetainedKVState, KEY, ENABLED, IDENTITY, BOUND, SCHEMA
from heterollm_sim.serving import (BatchCohort, BatchItem, BatchCost, RequestStatus,
    ServingRequest, _OnlineRuntime, compile_serving_plan)
from tests.test_nonflash_kv_view import scenario, SOURCE_SHA, KEY as VIEW_KEY, shape_tasks


def plan_for(parallel=2, *, requests=None, enabled=True, warmup=(512,32), batch=64, ubatch=64):
    case = scenario(parallel)
    if requests is None:
        requests = tuple(ServingRequest(f"r{i}", 0.0, 128, 2) for i in range(parallel))
    config = replace(case.llama_cpp_config, batch=batch, ubatch=ubatch)
    binding = dict(case.workload.metadata[VIEW_KEY]["configuration"], batch=batch, ubatch=ubatch)
    identity = {"process_id":"fixture-process", "process_block":"fixture-block", "runtime_build_id":"fixture-build"}
    raw = {"schema":SCHEMA, "identity":identity, "boundary":"after_final_qualified_warmup",
        "token_count_semantics":"native_prompt_including_bos_and_predicted_output",
        "warmup_batches":2, "complete_distinct_slots":True, "evidence_sha256":"a"*64,
        "source_sha256":SOURCE_SHA, "configuration":binding,
        "scope":{"completion_count":1,"singleton_owner":True,"ordinary_full_attention":True,"cache_prompt":False,
            "cache_ram_mib":0,"cache_idle_slots":False,"shared_prefix":False,"swa":False,
            "recurrent":False,"restore":False,"speculative":False,"external_state_operations":False,
            "context_shift":False,"purge":False,"cancellation":False,"recompute":False},
        "slots":[{"slot_id":i+10,"state":"retained","prompt_tokens":warmup[0],
            "output_tokens":warmup[1],"identity":identity} for i in range(parallel)],
        "request_slots":{r.request_id:10+i%parallel for i,r in enumerate(requests)}}
    case = replace(case, llama_cpp_config=config,
        placement=replace(case.placement, kv_policy=replace(case.placement.kv_policy, offload_ratio=0.0)),
        workload=replace(case.workload,
            scheduler=replace(case.workload.scheduler, max_num_batched_tokens=batch,
                max_num_ubatch_tokens=ubatch, preemption_enabled=False),
            metadata={**case.workload.metadata, VIEW_KEY:{**case.workload.metadata[VIEW_KEY], "configuration":binding},
                ENABLED:enabled, KEY:raw, IDENTITY:identity}))
    return replace(compile_serving_plan(case), requests=requests)


def ledger(plan):
    return RetainedKVState.from_plan(plan, tuple(planner._execution_layers(plan.scenario)))


def runtime(plan):
    return _OnlineRuntime(plan, lambda scenario,cohort: BatchCost(1.0))


def cohort(request="r0", phase="prefill", count=64, context=0):
    return BatchCohort("fixture", phase, 0.0, (BatchItem(request,phase,count,context),))


def admit(rt, request):
    rt._set_status(rt.states[request], RequestStatus.RUNNING)


@pytest.mark.parametrize("parallel",[1,2,4])
def test_admission_preserves_old_and_first_prompt_only_clears_selected_slot(parallel):
    rt = runtime(plan_for(parallel))
    state = rt._retained_kv_state
    assert sum(state.rows.values()) == parallel*543  # two warmups do not double count
    for request in rt.states:
        admit(rt,request)
    assert sum(state.rows.values()) == parallel*543
    rt._record_llama_engine_start(rt.states["r0"])
    assert sum(state.rows.values()) == (parallel-1)*543
    decorated = rt._with_kv_scan_lower_bound(cohort())
    assert decorated.metadata["llama_cpp_kv_occupied_rows"] == (parallel-1)*543+64
    assert state.rows[10] == 0  # preview is not a successful commit
    group, = planner._serving_invocation_groups(rt.plan.scenario,decorated)
    audit = group.nonflash_kv_view
    assert audit["occupied_cells_lower_bound"] == (parallel-1)*543+64
    assert audit["physical_k_tokens"] == ((max(1,(parallel-1)*543+64)+255)//256)*256
    assert audit["extent_completeness"] == "lower_bound"


def test_partial_prompt_commit_does_not_clear_again():
    state=ledger(plan_for())
    state.admit("r0")
    state.begin_prompt("r0")
    state.commit("r0","prefill",0,64)
    state.begin_prompt("r0")
    assert state.rows == {10:64,11:543}
    assert state.occupied_after((("r0","prefill",64,64),)) == 671


def test_finish_retains_then_next_request_admission_and_reuse_replace_once():
    reqs=(ServingRequest("r0",0.0,64,2),ServingRequest("next",1.0,32,1))
    rt=runtime(plan_for(1,requests=reqs))
    admit(rt,"r0")
    rt._record_llama_engine_start(rt.states["r0"])
    rt._execute(cohort(count=64))
    assert rt._retained_kv_state.rows == {10:64}
    rt._execute(cohort(phase="decode",count=1,context=64))
    assert rt.states["r0"].status == RequestStatus.FINISHED
    assert rt.states["r0"].kv_pages == 0
    assert rt._retained_kv_state.rows == {10:65}
    admit(rt,"next")
    assert rt._retained_kv_state.rows == {10:65}
    rt._record_llama_engine_start(rt.states["next"])
    rt._execute(cohort("next",count=32))
    assert rt._retained_kv_state.rows == {10:32}
    assert rt.states["next"].status == RequestStatus.FINISHED


def test_mixed_prefill_decode_and_finished_idle_slot_are_exclusive():
    state=RetainedKVState({10:543,11:543,12:543},{"a":10,"b":11},{"a":(128,32),"b":(64,1)},2048,6144)
    state.admit("a"); state.begin_prompt("a"); state.commit("a","prefill",0,128)
    state.admit("b"); state.begin_prompt("b")
    assert state.occupied_after((("a","decode",128,1),("b","prefill",0,63))) == 129+63+543
    state.commit("b","prefill",0,64); state.finish("b")
    assert state.occupied_after((("a","decode",128,1),)) == 129+64+543


def test_physical_groups_increment_from_cleared_base_not_final_cohort_rows():
    plan=plan_for(2,batch=128,ubatch=64)
    rt=runtime(plan)
    admit(rt,"r0"); rt._record_llama_engine_start(rt.states["r0"])
    decorated=rt._with_kv_scan_lower_bound(cohort(count=128))
    groups=planner._serving_invocation_groups(plan.scenario,decorated)
    assert [g.nonflash_kv_view["occupied_cells_lower_bound"] for g in groups] == [607,671]
    assert [g.nonflash_kv_view["physical_k_tokens"] for g in groups] == [768,768]


def test_only_prompt_candidates_with_budget_clear_old_slots():
    rt=runtime(plan_for(4))
    for request in rt.states: admit(rt,request)
    prepared=rt._prefill_cohort(list(rt.states.values()),False,token_budget=64)
    assert prepared.token_count == 64
    assert list(rt._retained_kv_state.rows.values()).count(0) == 1
    assert rt._with_kv_scan_lower_bound(prepared).metadata["llama_cpp_kv_occupied_rows"] == 1693


def test_failed_execution_never_commits_projected_kv():
    rt=runtime(plan_for())
    admit(rt,"r0"); rt._record_llama_engine_start(rt.states["r0"])
    rt.lowerer=lambda *args: (_ for _ in ()).throw(ValueError("fixture failure"))
    with pytest.raises(ValueError,match="fixture failure"): rt._execute(cohort())
    assert rt._retained_kv_state.rows == {10:0,11:543}


@pytest.mark.parametrize("operation",["unknown","purge","recompute","cancel","restore","shift"])
def test_unknown_lifecycle_invalidates_instead_of_retaining_stale_bound(operation):
    state=ledger(plan_for())
    with pytest.raises(ValueError,match="unsupported lifecycle"): state.invalidate(operation)
    with pytest.raises(ValueError,match="invalidated lifecycle"): state.occupied_after(())


def test_swap_rejected_at_runtime_transition():
    rt=runtime(plan_for())
    admit(rt,"r0")
    with pytest.raises(ValueError,match="swapped"): rt._set_status(rt.states["r0"],RequestStatus.SWAPPED)


@pytest.mark.parametrize("field",["slots","identity","scope","evidence_sha256","configuration","request_slots","source_sha256","warmup_batches","complete_distinct_slots","token_count_semantics"])
def test_missing_or_none_contract_fields_fail_closed(field):
    plan=plan_for()
    for missing in (True,False):
        raw=copy.deepcopy(plan.scenario.workload.metadata[KEY])
        if missing: raw.pop(field)
        else: raw[field]=None
        case=replace(plan.scenario,workload=replace(plan.scenario.workload,metadata={**plan.scenario.workload.metadata,KEY:raw}))
        with pytest.raises(ValueError): ledger(replace(plan,scenario=case))


@pytest.mark.parametrize("change",["duplicate","slot_identity","runtime_identity","unknown_slot","shared_prefix","purge","warmup_overflow"])
def test_bad_qualification_or_identity_is_rejected(change):
    plan=plan_for(); raw=copy.deepcopy(plan.scenario.workload.metadata[KEY])
    if change=="duplicate": raw["slots"][1]["slot_id"]=10
    elif change=="slot_identity": raw["slots"][1]["identity"]={"process_id":"different"}
    elif change=="runtime_identity": raw["identity"]={**raw["identity"],"process_id":"different"}
    elif change=="unknown_slot": raw["request_slots"]["r0"]=100
    elif change in {"shared_prefix","purge"}: raw["scope"][change]=True
    elif change=="warmup_overflow": raw["slots"][0]["prompt_tokens"]=2048
    case=replace(plan.scenario,workload=replace(plan.scenario.workload,metadata={**plan.scenario.workload.metadata,KEY:raw}))
    with pytest.raises(ValueError): ledger(replace(plan,scenario=case))


def test_hybrid_rejected_structurally_without_model_name_patches():
    plan=plan_for()
    hybrid=scenario(architecture="qwen3_5_hybrid_transformer")
    with pytest.raises(ValueError,match="hybrid/recurrent"):
        RetainedKVState.from_plan(plan,tuple(planner._execution_layers(hybrid)))


def test_sum_capacity_and_double_counting_fail_closed():
    with pytest.raises(ValueError,match="native capacity"):
        RetainedKVState({1:400,2:400},{},{},512,512)
    state=ledger(plan_for()); state.admit("r0"); state.begin_prompt("r0")
    with pytest.raises(ValueError,match="duplicate"):
        state.occupied_after((("r0","prefill",0,64),("r0","prefill",0,64)))
    with pytest.raises(ValueError,match="context mismatch"):
        state.commit("r0","prefill",543,64)
    with pytest.raises(ValueError,match="slot capacity"):
        state.commit("r0","prefill",0,2049)


def test_default_off_does_not_consume_invalid_contract_or_change_graph():
    plan=plan_for(enabled=False)
    raw=dict(plan.scenario.workload.metadata); raw[KEY]=None
    plan=replace(plan,scenario=replace(plan.scenario,workload=replace(plan.scenario.workload,metadata=raw)))
    rt=runtime(plan)
    assert rt._retained_kv_state is None
    original=cohort()
    assert rt._with_kv_scan_lower_bound(original) is original
    baseline=replace(plan.scenario,workload=replace(plan.scenario.workload,metadata={k:v for k,v in raw.items() if k not in {KEY,IDENTITY,ENABLED}}))
    assert planner.compile_serving_cohort_schedule(plan.scenario,original).tasks == planner.compile_serving_cohort_schedule(baseline,original).tasks


def test_enabled_missing_contract_is_rejected():
    plan=plan_for(); raw=dict(plan.scenario.workload.metadata); raw.pop(KEY)
    plan=replace(plan,scenario=replace(plan.scenario,workload=replace(plan.scenario.workload,metadata=raw)))
    with pytest.raises(ValueError,match="contract"): runtime(plan)


def test_occupied_metadata_and_template_cache_follow_retained_state():
    plan=plan_for()
    rt=runtime(plan)
    admit(rt,"r0");rt._record_llama_engine_start(rt.states["r0"])
    warm=rt._with_kv_scan_lower_bound(cohort())
    admit(rt,"r1");rt._record_llama_engine_start(rt.states["r1"])
    cleared=rt._with_kv_scan_lower_bound(cohort())
    assert planner._serving_cohort_cache_key(warm) != planner._serving_cohort_cache_key(cleared)
    context=planner.CompilationContext(plan.scenario)
    with planner._compilation_scope(plan.scenario,context):
        first=planner.compile_serving_cohort_schedule(plan.scenario,warm)
        second=planner.compile_serving_cohort_schedule(plan.scenario,cleared)
    assert {t.metadata[VIEW_KEY]["physical_k_tokens"] for t in shape_tasks(first)} == {768}
    assert {t.metadata[VIEW_KEY]["physical_k_tokens"] for t in shape_tasks(second)} == {256}
    assert second == planner.compile_serving_cohort_schedule(plan.scenario,cleared)


def test_same_padding_cached_graph_refreshes_occupancy_audit():
    plan=plan_for()
    rt=runtime(plan)
    admit(rt,"r0"); rt._record_llama_engine_start(rt.states["r0"])
    first=rt._with_kv_scan_lower_bound(cohort())
    second=replace(first,metadata={**first.metadata,"llama_cpp_kv_occupied_rows":700})
    context=planner.CompilationContext(plan.scenario)
    with planner._compilation_scope(plan.scenario,context):
        planner.compile_serving_cohort_schedule(plan.scenario,first)
        actual=planner.compile_serving_cohort_schedule(plan.scenario,second)
    assert {t.metadata[VIEW_KEY]["occupied_cells_lower_bound"] for t in shape_tasks(actual)} == {700}
    assert actual == planner.compile_serving_cohort_schedule(plan.scenario,second)


def test_explicit_assignment_cannot_admit_two_live_requests_to_same_slot():
    state=RetainedKVState({10:543},{"a":10,"b":10},{"a":(64,1),"b":(64,1)},2048,2048)
    state.admit("a")
    with pytest.raises(ValueError,match="still active"): state.admit("b")
    assert state.rows[10] == 543


def test_prompt_preparation_clears_even_if_logical_allocation_fails():
    rt=runtime(plan_for())
    admit(rt,"r0")
    rt._reserve_with_pressure=lambda *args:False
    assert rt._prefill_cohort([rt.states["r0"]],False) is None
    assert rt._retained_kv_state.rows == {10:0,11:543}


def test_planner_rejects_overcapacity_instead_of_clamping():
    plan=plan_for()
    decorated=replace(cohort(),metadata={BOUND:SCHEMA,"llama_cpp_kv_occupied_rows":4097})
    with pytest.raises(ValueError,match="capacity/context"):
        planner._serving_invocation_groups(plan.scenario,decorated)


def test_nonordinary_cohort_and_unprepared_rows_are_rejected():
    rt=runtime(plan_for())
    admit(rt,"r0")
    with pytest.raises(ValueError,match="before prompt clear"):
        rt._with_kv_scan_lower_bound(cohort())
    with pytest.raises(ValueError,match="nonordinary cohort"):
        rt._with_kv_scan_lower_bound(cohort(phase="recompute"))


def test_unmodeled_running_to_waiting_transition_invalidates_state():
    rt=runtime(plan_for())
    admit(rt,"r0")
    with pytest.raises(ValueError,match="waiting"):
        rt._set_status(rt.states["r0"],RequestStatus.WAITING)


def test_planner_requires_source_view_coverage_before_using_retained_sum():
    plan=plan_for()
    rt=runtime(plan)
    admit(rt,"r0"); rt._record_llama_engine_start(rt.states["r0"])
    decorated=rt._with_kv_scan_lower_bound(cohort())
    source={**plan.scenario.workload.metadata[VIEW_KEY], "runtime_binding_status":"unknown"}
    case=replace(plan.scenario,workload=replace(plan.scenario.workload,
        metadata={**plan.scenario.workload.metadata,VIEW_KEY:source}))
    with pytest.raises(ValueError,match="source/view not covered"):
        planner._serving_invocation_groups(case,decorated)


@pytest.mark.parametrize("failure_stage", ["lowerer", "commit"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_execution_or_commit_failure_poison_retained_state_and_reraise_original(
    failure_stage, error_type,
):
    rt = runtime(plan_for())
    admit(rt, "r0")
    rt._record_llama_engine_start(rt.states["r0"])
    rt._execute(cohort())
    retained = rt._retained_kv_state
    assert retained.rows == {10: 64, 11: 543}
    original_error = error_type("fixture execution failure")

    def fail(*args):
        raise original_error

    if failure_stage == "lowerer":
        rt.lowerer = fail
    else:
        successful_commit = retained.commit

        def partial_commit_then_fail(*args):
            successful_commit(*args)
            raise original_error

        retained.commit = partial_commit_then_fail
    retry = cohort(context=64)
    with pytest.raises(error_type) as caught:
        rt._execute(retry)
    assert caught.value is original_error
    assert retained.rows == {10: 64 if failure_stage == "lowerer" else 128, 11: 543}
    assert retained.invalid_reason == "cohort execution or commit failed: " + error_type.__name__
    with pytest.raises(ValueError, match="invalidated lifecycle"):
        rt._with_kv_scan_lower_bound(retry)
    with pytest.raises(ValueError, match="invalidated lifecycle"):
        admit(rt, "r1")
    with pytest.raises(ValueError, match="invalidated lifecycle"):
        rt._record_llama_engine_start(rt.states["r0"])
    with pytest.raises(ValueError, match="invalidated lifecycle"):
        rt._execute(retry)
    assert retained.invalid_reason == "cohort execution or commit failed: " + error_type.__name__


def test_default_off_preserves_exception_and_existing_retry_behavior():
    rt = runtime(plan_for(enabled=False))
    admit(rt, "r0")
    rt._record_llama_engine_start(rt.states["r0"])
    rt._execute(cohort())
    original_error = RuntimeError("fixture legacy failure")

    def fail(*args):
        raise original_error

    rt.lowerer = fail
    retry = cohort(context=64)
    with pytest.raises(RuntimeError) as caught:
        rt._execute(retry)
    assert caught.value is original_error
    assert rt._retained_kv_state is None
    assert rt._with_kv_scan_lower_bound(retry) is retry
    rt.lowerer = lambda scenario, item: BatchCost(1.0)
    rt._execute(retry)
    assert rt.states["r0"].prefill_cursor == 128
