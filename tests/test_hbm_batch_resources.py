"""Deferred publication preserves exact per-request bills and public state."""
import importlib.util
from pathlib import Path

import pytest

from heterollm_sim import _compiled_hbm, _hbm_pack
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import configuration
from test_hbm_task_resource_accumulator import assert_bill_equal, assert_state_equal, without_resources


@pytest.mark.parametrize("mapping", ("bank_interleaved", "row_local"))
@pytest.mark.parametrize("limit", (1, 3, 4096))
@pytest.mark.parametrize("scope", ("device", "channel"))
def test_owned_batches_match_public_requests_bit_for_bit(mapping, limit, scope):
    config = configuration(channels=8, address_mapping=mapping,
        interleave_bytes=128 if mapping == "bank_interleaved" else None,
        max_outstanding_requests=limit, queue_scope=scope,
        refresh_policy="all_bank", refresh_interval_ns=400., refresh_cycle_ns=40.)
    reference, actual = (DramCore(config, capture_details=False) for _ in range(2))
    reference._hbm_numeric_enabled = actual._hbm_numeric_enabled = True
    expected, accumulated = (_hbm_pack.HBMResourceAccumulator() for _ in range(2))
    span = config.row_bytes * config.bank_count if mapping == "row_local" else 128
    for core in (reference, actual):
        core.submit(AccessRequest("warm", "read", 0, 8192))
        # Public key order, existing bills and large Python integer totals
        # are all authoritative when the next batch begins.
        for field in ("busy_ns", "bytes_moved"):
            values = getattr(core.timeline, field)
            for key in tuple(values)[::2]:
                values[key] = values.pop(key)
    for accumulator in (expected, accumulated):
        accumulator.consume_metrics({"resource_bytes": {"dram:data:0": 2 ** 80},
            "resource_busy_ns": {"dram:data:0": 0.125}})
    for batch in range(3):
        requests = tuple(AccessRequest(f"{batch}:{index}", "read" if index % 3 else "write",
            (span * (index % config.lane_count) + 3) % (config.effective_capacity_bytes - 8192),
            (65, 2048, 8192)[index % 3], 1200. + batch * 700. + (index % 5) * 93.25)
            for index in range(65))
        packed = _compiled_hbm._pack_hbm_numeric(actual, requests[0], refresh_requests=requests)
        assert _compiled_hbm._can_batch_resources(actual, packed, accumulated)
        normal = reference.submit_many(requests)
        consumed = actual.submit_many(requests, _resource_accumulator=accumulated)
        for result in normal:
            expected.consume_metrics(result.counters)
        assert consumed == tuple(without_resources(result) for result in normal)
        assert_bill_equal(accumulated, expected)
        for field in ("ready_ns", "busy_ns", "last_intervals", "bytes_moved",
                      "lane_available", "directions", "lane_directions"):
            left, right = getattr(actual.timeline, field), getattr(reference.timeline, field)
            assert list(left) == list(right), field
            assert left == right, field
        assert actual._inflight == reference._inflight
        assert actual._channel_inflight == reference._channel_inflight
    assert_state_equal(actual, reference)


@pytest.mark.parametrize("failure", ("decline", "exception"))
def test_deferred_failure_publishes_successful_prefix_and_preserves_queue(failure, monkeypatch):
    config = configuration(channels=8, max_outstanding_requests=1)
    reference, actual = (DramCore(config, capture_details=False) for _ in range(2))
    reference._hbm_numeric_enabled = actual._hbm_numeric_enabled = True
    requests = (AccessRequest("prefix", "read", 0, 8192),
                AccessRequest("stop", "write", 65536, 8192),
                AccessRequest("tail", "read", 131072, 8192))
    expected, accumulated = (_hbm_pack.HBMResourceAccumulator() for _ in range(2))
    original = _compiled_hbm._HBMNumericState.execute
    stopped = set()
    def execute(packed, request):
        result = original(packed, request)
        if request.request_id == "stop" and id(packed.core) not in stopped:
            stopped.add(id(packed.core))
            if failure == "decline":
                return None
            raise RuntimeError("injected batch failure")
        return result
    monkeypatch.setattr(_compiled_hbm._HBMNumericState, "execute", execute)
    # Deliberately inject inside the closed executor for failure-path coverage.
    monkeypatch.setattr(_compiled_hbm, "_OWNED_TIMELINE_HBM_METHODS", tuple(
        (name, getattr(_compiled_hbm._HBMNumericState, name))
        for name, _ in _compiled_hbm._OWNED_TIMELINE_HBM_METHODS))
    if failure == "decline":
        normal = reference.submit_many(requests)
        assert actual.submit_many(requests, _resource_accumulator=accumulated) == tuple(
            without_resources(result) for result in normal)
    else:
        normal = reference.submit_many(requests[:1])
        with pytest.raises(RuntimeError, match="injected batch failure"):
            reference.submit(requests[1])
        with pytest.raises(RuntimeError, match="injected batch failure"):
            actual.submit_many(requests, _resource_accumulator=accumulated)
    for result in normal:
        expected.consume_metrics(result.counters)
    assert_bill_equal(accumulated, expected)
    assert_state_equal(actual, reference)


@pytest.mark.parametrize("source", (False, True))
def test_heap_admission_matches_unsorted_duplicate_public_credits(source):
    module = _hbm_pack
    if source:
        spec = importlib.util.spec_from_file_location("hbm_batch_python", Path(_hbm_pack.__file__).with_name("_hbm_pack.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    config = configuration(max_outstanding_requests=3)
    reference, actual = (DramCore(config, capture_details=False) for _ in range(2))
    reference._inflight = actual._inflight = [100., 20., 30., 20., 0.]
    admission = module.HBMAdmission(actual)
    for index, (arrival, completion) in enumerate(((0., 25.), (0., 27.), (90., 95.), (0., 100.), (200., 200.))):
        request = AccessRequest(str(index), "read", 0, 32, arrival)
        expected = reference._admission_time(request)
        ready = admission.admit(request)
        assert ready == expected
        reference._inflight = [end for end in reference._inflight if end > expected] + [completion]
        reference._acceptance_ns = actual._acceptance_ns = ready
        admission.record(completion, ready)
    admission.publish()
    assert actual._inflight == reference._inflight


@pytest.mark.parametrize("finish", ("commit", "rollback"))
def test_deferred_batches_keep_nested_runtime_transactions_and_external_calendars(finish):
    from heterollm_sim.data_motion import PhysicalRuntimeContext
    from test_hbm_resident_controller import visible
    config = configuration(channels=8, max_outstanding_requests=2)
    contexts = [PhysicalRuntimeContext(config, "hbm", capture_details=False) for _ in range(2)]
    reference, actual = [context.runtimes["hbm"].core for context in contexts]
    for core in (reference, actual):
        core.submit(AccessRequest("warm", "read", 0, 8192))
    outer = [context.begin_transaction(owners=("hbm",), _owned_timeline=True) for context in contexts]
    for iteration in range(2):
        expected, accumulated = (_hbm_pack.HBMResourceAccumulator() for _ in range(2))
        inner = [context.begin_transaction(owners=("hbm",), _owned_timeline=True) for context in contexts]
        for core in (reference, actual):
            rid = next(rid for rid in core.timeline.lane_available if ":data:" in rid)
            core.timeline.reserve(rid, core.timeline.available_ns(rid), 123.25)
        requests = tuple(AccessRequest(str(index), "read" if index % 2 else "write",
            index * 256, 8192, iteration * 1000.) for index in range(25))
        normal = reference.submit_many(requests)
        assert actual.submit_many(requests, _resource_accumulator=accumulated) == tuple(
            without_resources(result) for result in normal)
        for result in normal:
            expected.consume_metrics(result.counters)
        assert_bill_equal(accumulated, expected)
        for context, token in zip(contexts, inner):
            context.restore(token) if iteration == 0 else context.commit_transaction(token)
    for context, token in zip(contexts, outer):
        context.restore(token) if finish == "rollback" else context.commit_transaction(token)
    assert visible(contexts[0].snapshot()) == visible(contexts[1].snapshot())


def test_checkpoint_allocation_failure_keeps_deferred_successful_prefix(monkeypatch):
    reference, actual = (DramCore(configuration(channels=8), capture_details=False) for _ in range(2))
    reference._hbm_numeric_enabled = actual._hbm_numeric_enabled = True
    requests = (AccessRequest("prefix", "read", 0, 8192), AccessRequest("stop", "write", 65536, 8192))
    expected, accumulated = (_hbm_pack.HBMResourceAccumulator() for _ in range(2))
    normal = reference.submit_many(requests[:1])
    expected.consume_metrics(normal[0].counters)
    original = _compiled_hbm._HBMNumericState.checkpoint
    def checkpoint(packed, request):
        if request.request_id == "stop":
            raise MemoryError("injected checkpoint allocation failure")
        return original(packed, request)
    monkeypatch.setattr(_compiled_hbm._HBMNumericState, "checkpoint", checkpoint)
    monkeypatch.setattr(_compiled_hbm, "_OWNED_TIMELINE_HBM_METHODS", tuple(
        (name, getattr(_compiled_hbm._HBMNumericState, name))
        for name, _ in _compiled_hbm._OWNED_TIMELINE_HBM_METHODS))
    with pytest.raises(MemoryError, match="checkpoint allocation"):
        actual.submit_many(requests, _resource_accumulator=accumulated)
    assert_bill_equal(accumulated, expected)
    assert_state_equal(actual, reference)


def test_python_source_deferred_resources_matches_compiled_public_bills(monkeypatch):
    spec = importlib.util.spec_from_file_location("hbm_batch_python_resources", Path(_hbm_pack.__file__).with_name("_hbm_pack.py"))
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    monkeypatch.setattr(_compiled_hbm, "HBMResourceBatch", source.HBMResourceBatch)
    classes = (source.HBMResourceBatch, _compiled_hbm.HBMAdmission, _hbm_pack.HBMResourceAccumulator)
    monkeypatch.setattr(_compiled_hbm, "_OWNED_TIMELINE_BATCH_CLASSES", classes)
    monkeypatch.setattr(_compiled_hbm, "_OWNED_TIMELINE_BATCH_METHODS", tuple(
        (cls, name, getattr(cls, name)) for cls in classes
        for name, value in vars(cls).items() if callable(value)))
    reference, actual = (DramCore(configuration(channels=8), capture_details=False) for _ in range(2))
    reference._hbm_numeric_enabled = actual._hbm_numeric_enabled = True
    expected, accumulated = (_hbm_pack.HBMResourceAccumulator() for _ in range(2))
    requests = tuple(AccessRequest(str(index), "write" if index % 3 else "read",
                                  index * 256, 8192, index * 99.25) for index in range(32))
    packed = _compiled_hbm._pack_hbm_numeric(actual, requests[0], refresh_requests=requests)
    assert _compiled_hbm._can_batch_resources(actual, packed, accumulated)
    normal = reference.submit_many(requests)
    assert actual.submit_many(requests, _resource_accumulator=accumulated) == tuple(
        without_resources(result) for result in normal)
    for result in normal:
        expected.consume_metrics(result.counters)
    assert_bill_equal(accumulated, expected)
    assert_state_equal(actual, reference)
