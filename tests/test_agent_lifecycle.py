"""Real session inheritance and safe physical cancellation regressions."""
from dataclasses import replace
from pathlib import Path
import copy
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hbf_hbm_agent_study import build_agent_scenario
from heterollm_sim.kv_pool import KvPoolError
from heterollm_sim.reporting import run_scenario
from heterollm_sim.serving import RequestStatus, _OnlineRuntime, compile_serving_plan
from test_async_kv_prefetch import offloaded_runtime


def test_handoff_reuses_same_page_generation_and_physical_addresses():
    runtime, low = offloaded_runtime()
    pool = runtime.ledger.paged_pool
    executor = runtime._paged_pool_transfer_executor
    before = [(p.logical_page_id, p.allocation_generation, executor.address(p)) for p in pool.request_pages("low")]
    capacity_before = dict(runtime.physical_ledger.used_bytes)
    ids = pool.handoff_request("low", "next-turn")
    executor.sync_pages(pool.pages())
    after = [(p.logical_page_id, p.allocation_generation, executor.address(p)) for p in pool.request_pages("next-turn")]
    assert before == after
    assert ids == tuple(item[0] for item in before)
    assert not pool.request_pages("low")
    assert capacity_before == runtime.physical_ledger.used_bytes
    assert all(p.request_id == "next-turn" for p in pool.request_pages("next-turn"))
    with pytest.raises(KvPoolError, match="empty destination"):
        pool.handoff_request("high", "next-turn")


@pytest.mark.parametrize("after_target_write", [False, True])
def test_cancel_preserves_source_and_work_and_waits_for_submitted_write(after_target_write):
    runtime, low = offloaded_runtime(distance=1)
    pool = runtime.ledger.paged_pool
    executor = runtime._paged_pool_transfer_executor
    initial = dict(runtime.physical_ledger.used_bytes)
    assert runtime._prefetch_waiting()
    page_id, plan = runtime._pending_prefetches["low"]["prepared"][0]
    page = pool.page(page_id)
    identity = (page.owner_component, page.allocation_generation, executor.address(page))
    if after_target_write:
        while plan.advance is not None:
            plan = plan.advance()
    with pytest.raises(KvPoolError, match="drain"):
        pool.cancel_prepared_restore(page_id, plan, now_ns=plan.completion_ns - 1)
    assert runtime.physical_ledger.used_bytes["hbm0"] == initial["hbm0"] + page.bytes
    # An independent operation submitted after the copy must survive its cancel.
    executor.runtime.timeline.reserve("unrelated-operation", 100., 777., direction="read_write")
    timeline_before = copy.deepcopy(executor.runtime.timeline.__dict__)
    counters_before = dict(executor.counters)
    bill = pool.cancel_prepared_restore(page_id, plan, now_ns=plan.completion_ns)
    assert executor.runtime.timeline.__dict__ == timeline_before
    assert all(executor.counters[key] == value for key, value in counters_before.items())
    assert identity == (page.owner_component, page.allocation_generation, executor.address(page))
    assert runtime.physical_ledger.used_bytes == initial
    assert pool.cache_accesses("low", first_token=0, token_count=16)
    kinds = {phase["kind"] for phase in bill["phases"]}
    assert "source_read" in kinds
    assert ("target_write" in kinds) == after_target_write
    assert sum(phase.get("service_ns", 0.) for phase in bill["phases"]) > 0


def test_full_serving_computes_only_new_tokens_and_resumes_after_copy():
    result = run_scenario(build_agent_scenario()).serving
    assert all(metric.status is RequestStatus.FINISHED for metric in result.request_metrics.values())
    assert result.agent_lifecycle_metrics["handoffs"] == 2
    assert result.agent_lifecycle_metrics["reused_tokens"] == 154
    prefill = {request: sum(item.token_count for batch in result.batches for item in batch.items
        if item.request_id == request and item.phase == "prefill") for request in ("agent-0", "agent-1", "agent-2")}
    assert prefill == {"agent-0": 64, "agent-1": 17, "agent-2": 17}
    copy = next(event for event in result.events if event.event_type == "kv_prefetch")
    child_batches = [batch for batch in result.batches if "agent-1" in batch.request_ids]
    assert min(batch.start_ns for batch in child_batches) >= copy.timestamp_ns
    assert result.kv_metrics.prefetch_useful_bytes > 0
    assert result.runtime_kernel_metrics["paged_kv_pool"]["page_count"] == 0
    from hbf_hbm_metrics import prefetch_overlap
    overlap = prefetch_overlap(result, {"gpu0"})
    assert overlap["decode_operator_overlap_ns"] > 0
    assert overlap["premature_consumer_overlap_ns"] == 0
    # Turn arrival is relative to the actual prior completion, not t=0.
    assert result.request_states["agent-1"].arrival_ns == pytest.approx(
        result.request_metrics["agent-0"].finish_ns + 500000.)


def test_idle_ttl_cancels_inflight_copy_and_recomputes_entire_prompt():
    result = run_scenario(build_agent_scenario(ttl_ns=400000., think_times=(2000000.,),
        predictions=(100000.,), prefetch_lead_ns=100000., hbf_page_read_ns=500000.)).serving
    assert result.agent_lifecycle_metrics["cancellations"] == 1
    assert result.agent_lifecycle_metrics["expired_tokens"] == 67
    assert result.kv_metrics.prefetch_cancelled_bytes == 20480
    assert "drains submitted physical phases" in result.kv_metrics.prefetch_metric_semantics
    assert "cancellation is not modeled" not in result.kv_metrics.prefetch_metric_semantics
    assert result.kv_metrics.prefetch_inflight_bytes == 0
    assert result.agent_lifecycle_metrics["reused_tokens"] == 0
    child_prefill = sum(item.token_count for batch in result.batches for item in batch.items
                        if item.request_id == "agent-1" and item.phase == "prefill")
    assert child_prefill == 84
    cancelled = next(event for event in result.events if event.event_type == "kv_prefetch_cancelled")
    assert cancelled.timestamp_ns >= cancelled.details["requested_ns"]
    assert all(phase["completion_ns"] <= cancelled.timestamp_ns for phase in cancelled.details["submitted_phases"])
    assert result.runtime_kernel_metrics["paged_kv_pool"]["page_count"] == 0
    assert all(metric.status is RequestStatus.FINISHED for metric in result.request_metrics.values())


def test_rejects_implicit_branching_instead_of_double_owning_pages():
    scenario = build_agent_scenario(background=False)
    root, child, _ = scenario.workload.requests
    sibling = replace(child, request_id="sibling")
    scenario = replace(scenario, workload=replace(scenario.workload, requests=(root, child, sibling)))
    with pytest.raises(ValueError, match="single child"):
        _OnlineRuntime(compile_serving_plan(scenario), lambda *_: 1.)
