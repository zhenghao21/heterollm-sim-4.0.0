"""Native request accounting preserves ordered float bits and host admission."""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from random import Random

import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_types import AccessRequest
from test_generic_nand_resident import logical
from test_nand_generic_buffered_acceleration import PAGE, config
from test_token_owned_history import bits


def numerical_bits(value):
    # The pre-existing bulk publisher and the scalar machine insert the
    # initial resource maps in different orders. Compare every float's hex
    # representation, keeping list/queue order and treating maps as maps.
    # test_generic_nand_resident separately checks bulk-publication order.
    if isinstance(value, Mapping):
        return tuple(sorted(((key, numerical_bits(item)) for key, item in value.items()), key=lambda row: repr(row[0])))
    if is_dataclass(value):
        return type(value), tuple((field.name, numerical_bits(getattr(value, field.name))) for field in fields(value))
    if isinstance(value, (list, tuple)):
        return type(value), tuple(numerical_bits(item) for item in value)
    return bits(value)


def contexts(settings):
    result = [PhysicalRuntimeContext(settings, "flash", capture_details=False) for _ in range(2)]
    scalar = result[1].runtimes["flash"].core
    scalar._nand_numeric_enabled = False
    scalar._execute_accepted = lambda request: scalar._execute_detailed_accepted(request)
    return result


def compare(pair, request):
    actual, expected = [context.runtimes["flash"].core.submit(request) for context in pair]
    counters = dict(actual.counters)
    native_pages = counters.pop("native_fullpage_pages", None)
    assert numerical_bits(replace(actual, counters=counters)) == numerical_bits(expected)
    assert numerical_bits(logical(pair[0].runtimes["flash"].core)) == numerical_bits(logical(pair[1].runtimes["flash"].core))
    return actual, native_pages


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("duplex", [False, True])
def test_request_sequence_partial_rmw_hits_and_admission_are_bit_exact(owned, depth, duplex):
    pair = contexts(config(max_outstanding_requests=depth, max_outstanding_write_requests=1,
                           host_full_duplex=duplex, host_channels=2, host_granularity_bytes=256))
    compare(pair, AccessRequest("seed", "read", 0, 128 * PAGE, .125))
    tokens = [context.begin_transaction(owners=("flash",), _owned_timeline=owned and index == 0)
              for index, context in enumerate(pair)]
    rng = Random(1097)
    waited = False
    # Same-arrival read batches include all-cache-hit requests; mixed one-page
    # appends preserve write limits, old-page RMW and cache reader lifetimes.
    operations = [("read", 0, 128 * PAGE), ("read", 0, 128 * PAGE),
                  ("write", 17, 99), ("read", 17, 99),
                  ("write", PAGE - 7, 20), ("read", PAGE - 7, 20)]
    operations += [("write" if index % 3 == 0 else "read",
                    rng.randrange(300) * PAGE + rng.randrange(1000),
                    rng.randrange(1, 150) * PAGE - rng.randrange(1000)) for index in range(24)]
    for index, (op, address, size) in enumerate(operations):
        row, native = compare(pair, AccessRequest(str(index), op, address, size, .125))
        span = (address + size - 1) // PAGE - address // PAGE + 1
        assert native == (span if span >= 2 else None)
        waited |= row.counters["queue_wait_ns"] > 0
    assert waited
    for context, token in zip(pair, tokens):
        context.restore(token)
    assert numerical_bits(logical(pair[0].runtimes["flash"].core)) == numerical_bits(logical(pair[1].runtimes["flash"].core))


@pytest.mark.parametrize("finish", ["commit_transaction", "restore"])
def test_nested_owned_request_accounting_restores_every_cumulative_counter(finish):
    pair = contexts(config(max_outstanding_requests=2))
    compare(pair, AccessRequest("seed", "read", 0, 128 * PAGE))
    before = numerical_bits(logical(pair[0].runtimes["flash"].core))
    outer = [context.begin_transaction(owners=("flash",), _owned_timeline=index == 0)
             for index, context in enumerate(pair)]
    compare(pair, AccessRequest("prefix", "write", 33, 128 * PAGE + 7))
    inner = [context.begin_transaction(owners=("flash",), _owned_timeline=index == 0)
             for index, context in enumerate(pair)]
    for index in range(8):
        compare(pair, AccessRequest(str(index), "read", index * PAGE + 17, 128 * PAGE))
    for context, token in zip(pair, inner):
        getattr(context, finish)(token)
    compare(pair, AccessRequest("tail", "write", 17, 99))
    for context, token in zip(pair, outer):
        context.restore(token)
    assert numerical_bits(logical(pair[0].runtimes["flash"].core)) == before
    assert numerical_bits(logical(pair[1].runtimes["flash"].core)) == before


def test_native_summary_multilane_and_shared_array_slots_keep_exact_metrics():
    pair = contexts(config(parallel_units=2, host_channels=2))
    for context in pair:
        context.timeline.lane_available.update({"nand:parallel:0": [0., 77.125],
                                                "nand:host:0:read": [12.5, 0.]})
    for index, op in enumerate(("read", "read", "write", "read")):
        compare(pair, AccessRequest(str(index), op, index * PAGE + 33, 128 * PAGE, .125))


def test_invalid_request_after_native_batch_retains_the_original_partial_state():
    pair = contexts(config())
    compare(pair, AccessRequest("seed", "read", 0, 128 * PAGE))
    compare(pair, AccessRequest("prefix", "write", 17, 99))
    for context in pair:
        with pytest.raises(ValueError):
            context.runtimes["flash"].core.submit(AccessRequest("invalid", "read",
                context.config.effective_capacity_bytes, PAGE))
    assert numerical_bits(logical(pair[0].runtimes["flash"].core)) == numerical_bits(logical(pair[1].runtimes["flash"].core))


def test_authored_undo_writer_retains_original_request_accounting(monkeypatch):
    from heterollm_sim.memory_transfer import _OwnedTimelineTransaction

    pair = contexts(config())
    compare(pair, AccessRequest("seed", "read", 0, 128 * PAGE))
    tokens = [context.begin_transaction(owners=("flash",), _owned_timeline=True) for context in pair]
    original = _OwnedTimelineTransaction.remember
    accounting = []

    def authored(token, resource):
        accounting.append(token.accounting)
        return original(token, resource)

    monkeypatch.setattr(_OwnedTimelineTransaction, "remember", authored)
    compare(pair, AccessRequest("authored", "read", 17, 128 * PAGE))
    assert any(accounting)
    for context, token in zip(pair, tokens):
        context.restore(token)
    assert numerical_bits(logical(pair[0].runtimes["flash"].core)) == numerical_bits(logical(pair[1].runtimes["flash"].core))


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("multilane", [False, True])
def test_submit_many_keeps_each_request_bits_admission_and_rollback(depth, multilane):
    settings = config(channels=112, planes_per_lun=25, max_outstanding_requests=depth,
                      host_granularity_bytes=256, parallel_units=2 if multilane else None)
    pair = contexts(settings)
    sequential = PhysicalRuntimeContext(settings, "flash", capture_details=False)
    if multilane:
        for context in (*pair, sequential):
            context.timeline.lane_available.update({"flash:nand:parallel:0": [0., 77.125],
                                                    "flash:nand:host:read": [12.5, 0.]})
    seed = AccessRequest("seed", "read", 0, 128 * PAGE)
    compare(pair, seed)
    sequential.runtimes["flash"].core.submit(seed)
    all_contexts = [*pair, sequential]
    before = [numerical_bits(logical(context.runtimes["flash"].core)) for context in all_contexts]
    tokens = [context.begin_transaction(owners=("flash",), _owned_timeline=True) for context in all_contexts]
    requests = [AccessRequest(str(index), "read", (index % 15) * 117 * PAGE + 17,
                              116 * PAGE - 23, .125) for index in range(40)]
    fast = pair[0].runtimes["flash"].core
    actual = fast.submit_many(requests)
    reference = [pair[1].runtimes["flash"].core.submit(request) for request in requests]
    ordered = [sequential.runtimes["flash"].core.submit(request) for request in requests]
    for a, b, c in zip(actual, reference, ordered):
        counters = dict(a.counters)
        counters.pop("native_fullpage_pages")
        assert numerical_bits(replace(a, counters=counters)) == numerical_bits(b)
        assert bits(a) == bits(c)
    assert "_nand_request_metric_orders" not in fast.__dict__
    assert bits(logical(fast)) == bits(logical(sequential.runtimes["flash"].core))
    for context, token, saved in zip(all_contexts, tokens, before):
        context.restore(token)
        assert numerical_bits(logical(context.runtimes["flash"].core)) == saved


def test_submit_many_invalid_suffix_executes_the_same_valid_prefix():
    pair = contexts(config())
    compare(pair, AccessRequest("seed", "read", 0, 128 * PAGE))
    requests = [AccessRequest("valid", "read", 17, 128 * PAGE),
                AccessRequest("invalid", "read", pair[0].config.effective_capacity_bytes, PAGE)]
    for index, context in enumerate(pair):
        core = context.runtimes["flash"].core
        with pytest.raises(ValueError):
            if index == 0:
                core.submit_many(requests)
            else:
                for request in requests:
                    core.submit(request)
        assert "_nand_request_metric_orders" not in core.__dict__
    assert numerical_bits(logical(pair[0].runtimes["flash"].core)) == numerical_bits(logical(pair[1].runtimes["flash"].core))
