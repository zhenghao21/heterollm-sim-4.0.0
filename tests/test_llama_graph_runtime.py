"""Tiny real-provider integration plus pure lifecycle transaction regressions."""
from dataclasses import replace
from types import SimpleNamespace
import copy

import pytest
from heterollm_sim import serving
from heterollm_sim import llama_graph_runtime as gr
from heterollm_sim.ir import RequestSpec
from heterollm_sim.llama_scenario import apply_llama_runtime_config
from heterollm_sim.planner import TopologyAwareBatchCostProvider
from heterollm_sim.reporting import run_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
from tests.test_llama_mixed_batching import authored


def declared_scope():
    contract={"schema":"llama.cpp.mixed-phase-batching/v1", "status":"enabled",
        "reason":"ordinary_dense_full_attention_text", "graph_qualified":True,
        "source":"synthetic source-derived adapter fixture",
        "evidence_kind":"source_derived_execution_semantics"}
    return SimpleNamespace(llama_cpp_config=SimpleNamespace(kv_unified=True),
        workload=SimpleNamespace(mtp=None,metadata={"llama_cpp_mixed_phase_batching":contract}))


def cohort(ident, n=4):
    return serving.BatchCohort(ident,"prefill",0.0,
        (serving.BatchItem("request", "prefill", n, 0, logit_tokens=1),))


def groups(n=4, outputs=1):
    return {"operator_invocation_group_count":1,"operator_invocation_groups":({
        "group_id":"operator-invocation-group-0000", "group_index":0,
        "token_batch":n,"physical_ubatch_rows":n,"logit_token_batch":outputs,
        "request_ids":("request",),"batching_semantics":"stateless_scheduler_batch",
        "lanes":tuple({"request_id":"request","requires_logits":i>=n-outputs} for i in range(n)),
        # A lower bound must never become an exact graph input comparison.
        "llama_cpp_nonflash_kv_view":{"applied":True,"physical_k_tokens":256,"extent_completeness":"lower_bound"},
    },)}


def test_unknown_initial_and_transaction_commit_only():
    recorder=gr.LlamaGraphRuntime(declared_scope())
    initial=recorder.state
    pending=recorder.prepare(cohort("first"),groups())
    assert recorder.state is initial and recorder.summary()["observed_ubatch_transition_count"]==0
    assert pending.records[0]["decision"]=="unknown"
    assert "previous_graph_unknown" in pending.records[0]["reasons"]
    recorder.commit(pending)
    assert recorder.summary()["successful_cohort_count"]==1
    with pytest.raises(ValueError,match="stale"):recorder.commit(pending)


def test_A_B_A_tracks_previous_context_not_shape_cache():
    recorder=gr.LlamaGraphRuntime(declared_scope())
    for index,n in enumerate((4,2,4)):
        recorder.commit(recorder.prepare(cohort(str(index),n),groups(n)))
    rows=recorder.summary()["transitions"]
    assert [r["decision"] for r in rows]==["unknown","miss","miss"]
    assert "parameter_changed:n_tokens" in rows[2]["reasons"]
    assert [r["generation_after"] for r in rows]==[1,2,3]
    assert len({(r["cohort_id"],r["group_index"]) for r in rows})==3


def test_padded_lower_bound_and_unknown_roster_never_prove_hit():
    recorder=gr.LlamaGraphRuntime(declared_scope())
    for ident in ("one","two"):
        recorder.commit(recorder.prepare(cohort(ident),groups()))
    row=recorder.summary()["transitions"][-1]
    assert row["decision"]=="unknown"
    assert row["parameters_known"]=={"equal_seqs":False,"n_tokens":4,"n_seq_tokens":1,"n_seqs":4,"n_outputs":1}
    assert "cached_input_roster_unknown" in row["reasons"]
    assert not any("mask" in key for key in row["parameters_known"])
    assert row["output_row_indices"]==(3,) and row["priced"] is False
    summary=recorder.summary()
    assert summary["service_cost_ns"] is None and summary["duration_adjustment_ns"]==0
    assert summary["qualification"]["native_execution_verified"] is False


def test_failure_invalidates_without_committing_prepared_ubatches():
    recorder=gr.LlamaGraphRuntime(declared_scope())
    pending=recorder.prepare(cohort("failure"),groups())
    recorder.failed("failure",RuntimeError("synthetic downstream failure"))
    assert recorder.state.previous_unknown and recorder.state.previous is None
    assert recorder.summary()["observed_ubatch_transition_count"]==0
    assert recorder.summary()["failed_cohort_count"]==1
    with pytest.raises(ValueError,match="stale"):recorder.commit(pending)


@pytest.mark.parametrize("change",["non_llama","missing_contract","missing_groups","bad_count","bad_lanes","bad_outputs"])
def test_unsupported_or_incomplete_is_unobserved_not_a_fake_transition(change):
    scenario=declared_scope();data=groups()
    if change=="non_llama":scenario.llama_cpp_config=None
    elif change=="missing_contract":scenario.workload.metadata={}
    elif change=="missing_groups":data={}
    elif change=="bad_count":data["operator_invocation_group_count"]=2
    else:
        data=copy.deepcopy(data);row=data["operator_invocation_groups"][0]
        if change=="bad_lanes":row["lanes"]=row["lanes"][:-1]
        else:row["logit_token_batch"]=2
    recorder=gr.LlamaGraphRuntime(scenario)
    recorder.commit(recorder.prepare(cohort("unobserved"),data))
    result=recorder.summary()
    assert result["status"]=="unobserved" and result["observed_ubatch_transition_count"]==0
    assert result["unobserved_cohort_count"]==1 and recorder.state.previous_unknown


def test_bounded_records_do_not_shrink_transition_denominator(monkeypatch):
    monkeypatch.setattr(gr,"RECORD_LIMIT",2)
    recorder=gr.LlamaGraphRuntime(declared_scope())
    for index in range(4):recorder.commit(recorder.prepare(cohort(str(index)),groups()))
    result=recorder.summary()
    assert result["observed_ubatch_transition_count"]==4 and result["retained_transition_count"]==2
    assert result["dropped_transition_count"]==2 and result["all_observed_records_retained"] is False


def tiny_scenario():
    case=authored(3)
    workload=replace(case.workload,requests=(RequestSpec("a",0.0,8,3),RequestSpec("b",0.0,12,2)))
    return apply_llama_runtime_config(replace(case,workload=workload),
        LlamaCppRuntimeConfig(batch=8,ubatch=4,context=64,parallel=2,cont_batching=True))


def assert_real_diagnostics(result):
    batches=[b for b in result.serving.batches if b.kind in {"prefill","decode","mixed"}]
    summary=result.serving.runtime_kernel_metrics["llama_cpu_graph_lifecycle"]
    expected=sum(len(b.cost.metadata["operator_invocation_groups"]) for b in batches)
    assert summary["successful_cohort_count"]==len(batches)
    assert summary["observed_cohort_count"]==len(batches)
    assert summary["observed_ubatch_transition_count"]==expected
    assert summary["unobserved_cohort_count"]==0 and expected>len(batches)
    rows=summary["transitions"]
    assert len(rows)==expected and len({(r["cohort_id"],r["group_index"]) for r in rows})==expected
    assert rows[0]["decision"]=="unknown" and summary["native_reuse_verified"] is False
    assert all(r["priced"] is False for r in rows)


def test_real_provider_cache_replay_still_records_every_actual_ubatch_and_preserves_time(monkeypatch):
    scenario=tiny_scenario();provider=TopologyAwareBatchCostProvider(scenario)
    first=run_scenario(scenario,batch_lowerer=provider)
    assert_real_diagnostics(first)
    hits=provider.template_cache_stats["hits"]
    second=run_scenario(scenario,batch_lowerer=provider)
    assert provider.template_cache_stats["hits"]>hits
    assert_real_diagnostics(second)
    assert second.serving.request_metrics==first.serving.request_metrics
    assert second.serving.batches==first.serving.batches
    assert second.serving.events==first.serving.events
    # Disable only the diagnostic recorder to compare against the unchanged
    # real planner/provider/runtime, not against a constant synthetic cost.
    class NoopRecorder:
        def __init__(self,scenario):pass
        def prepare(self,*a):return None
        def commit(self,*a):pass
        def failed(self,*a):pass
        def summary(self):return {"status":"disabled_by_test"}
    monkeypatch.setattr(serving,"LlamaGraphRuntime",NoopRecorder)
    baseline=run_scenario(scenario,batch_lowerer=provider)
    assert baseline.serving.makespan_ns==first.serving.makespan_ns
    assert baseline.serving.request_metrics==first.serving.request_metrics
    assert baseline.serving.batches==first.serving.batches
    assert baseline.serving.events==first.serving.events


def test_real_execute_failure_records_no_success_and_leaves_unknown(monkeypatch):
    captured=[]
    class ObservedRecorder(gr.LlamaGraphRuntime):
        def __init__(self,scenario):
            super().__init__(scenario);captured.append(self)
    monkeypatch.setattr(serving,"LlamaGraphRuntime",ObservedRecorder)
    def fail_after_lowering(self,cohort,cost):raise RuntimeError("synthetic actual execution failure")
    monkeypatch.setattr(serving._OnlineRuntime,"_apply_resource_contention",fail_after_lowering)
    with pytest.raises(RuntimeError,match="actual execution failure"):run_scenario(tiny_scenario())
    assert len(captured)==1
    summary=captured[0].summary()
    assert summary["failed_cohort_count"]==1 and summary["successful_cohort_count"]==0
    assert summary["observed_ubatch_transition_count"]==0 and captured[0].state.previous_unknown
