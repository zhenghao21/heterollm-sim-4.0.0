"""Report readers use the current serving-result fields without old objects."""
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest

from heterollm_sim.reporting import (
    OnlineScenarioResult, _online_generation_subtargets, _online_mtp_draft_subtargets,
    _online_subtarget_request_rows, _sim_hardware_mtp_rows, _sim_hardware_phase_ledger,
    _sim_subtarget_batch_phases, report_dict,
)
from heterollm_sim.serving import (
    BatchCost, BatchItem, OwnerResidencyMetrics, RequestStatus, ServingBatch, ServingEvent,
)


@pytest.fixture
def batch():
    item = BatchItem("request", "mtp", 3, 8, main_tokens=1, draft_tokens=2,
                     verifier_tokens=3, committed_tokens=2, logit_tokens=3)
    return ServingBatch("cohort", "mtp", 10., 20., ("request",), 3,
                        BatchCost(10.), (item,))


def missing_field(value, field):
    return SimpleNamespace(**{
        item.name: getattr(value, item.name) for item in fields(value) if item.name != field
    })


def test_current_mtp_items_and_empty_transfer_batches_have_distinct_meanings(batch):
    assert _sim_subtarget_batch_phases(batch) == {"mtp"}
    rows, complete = _sim_hardware_mtp_rows(batch)
    assert complete
    assert rows == {"main_rows": 1, "draft_rows": 2, "committed_rows": 2, "rejected_rows": 1}
    transfer = replace(batch, kind="kv_swap_out", items=(), token_count=0)
    assert _sim_subtarget_batch_phases(transfer) == {"kv_swap_out"}
    assert _sim_hardware_mtp_rows(transfer) == ({key: 0 for key in rows}, True)


@pytest.mark.parametrize("field", ["items", "request_ids"])
def test_request_report_rejects_old_batches_missing_required_fields(batch, field):
    serving = SimpleNamespace(batches=(missing_field(batch, field),), request_metrics={})
    with pytest.raises(AttributeError, match=field):
        _online_subtarget_request_rows(serving)


def test_generation_report_requires_current_batch_identity(batch):
    serving = SimpleNamespace(batches=(missing_field(batch, "cohort_id"),),
                              events=(), request_metrics={})
    with pytest.raises(AttributeError, match="cohort_id"):
        _online_generation_subtargets(serving)


@pytest.mark.parametrize("field", ["phase", "main_tokens", "draft_tokens", "committed_tokens"])
def test_mtp_report_does_not_default_old_item_fields(batch, field):
    old = replace(batch, items=(missing_field(batch.items[0], field),))
    with pytest.raises(AttributeError, match=field):
        _sim_hardware_mtp_rows(old)


@pytest.mark.parametrize("reader", [_online_generation_subtargets, _online_mtp_draft_subtargets])
@pytest.mark.parametrize("field", ["event_type", "request_id"])
def test_token_report_requires_current_event_shape(batch, reader, field):
    event = ServingEvent(20., "tokens_committed", "request", "cohort", {"visible_tokens": 2})
    serving = SimpleNamespace(batches=(batch,), events=(missing_field(event, field),),
        request_metrics={"request": SimpleNamespace(status=RequestStatus.FINISHED)})
    with pytest.raises(AttributeError, match=field):
        reader(serving)


def test_report_keeps_explicit_truncation_and_optional_residency(batch):
    truncated = replace(batch, items=(), metadata={"items_truncated": True})
    _, diagnostics = _online_subtarget_request_rows(SimpleNamespace(
        batches=(truncated,), request_metrics={}))
    assert diagnostics["batch_item_coverage_complete"] is False
    assert diagnostics["truncated_batches"] == 1
    without_owner = _sim_hardware_phase_ledger(())
    assert without_owner == _sim_hardware_phase_ledger((), owner_residency_metrics=OwnerResidencyMetrics())
    with pytest.raises(AttributeError, match="enabled"):
        _sim_hardware_phase_ledger((), owner_residency_metrics=SimpleNamespace())


def test_current_online_result_reports_through_serving_without_old_aliases():
    from test_async_kv_prefetch import runtime
    from test_backend_execution_semantics import manifest

    runner = runtime()
    result = OnlineScenarioResult(runner.plan.scenario, runner.run(), manifest(), (), "aggregate")
    report = report_dict(result)
    assert report["summary"]["makespan_ns"] == result.serving.makespan_ns
    for removed in ("trace", "metrics"):
        with pytest.raises(AttributeError):
            getattr(result, removed)
