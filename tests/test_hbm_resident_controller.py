"""Resident arrays retain the externally visible controller and undo contract."""
import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import configuration


def runtime_pair():
    fast = PhysicalRuntimeContext(configuration(), "hbm", capture_details=False)
    reference = PhysicalRuntimeContext(configuration(), "hbm", capture_details=False)
    reference.runtimes["hbm"].core._hbm_numeric_enabled = False
    return fast, reference


def visible(snapshot):
    return (snapshot["runtime_state"]["hbm"]["core"], snapshot["timeline"],
            snapshot["allocator_state"], snapshot["committed_owners"])


def test_resident_mixed_submissions_and_batches_match_every_metric_and_published_state():
    fast, reference = runtime_pair()
    cores = [runtime.runtimes["hbm"].core for runtime in (fast, reference)]
    for number in range(10):
        transaction = fast.begin_transaction()
        requests = [AccessRequest(f"{number}-{index}", "write" if number % 3 == 0 else "read",
            (number * 131072 + index * 8192) % 524288, 8192, number * 1000.) for index in range(3)]
        actual = cores[0].submit_many(requests) if number % 2 else tuple(cores[0].submit(item) for item in requests)
        expected = cores[1].submit_many(requests) if number % 2 else tuple(cores[1].submit(item) for item in requests)
        assert actual == expected
        fast.commit_transaction(transaction)
        assert cores[0]._hbm_numeric_state is not None
    assert visible(fast.snapshot()) == visible(reference.snapshot())
    assert cores[0]._hbm_numeric_state is None


@pytest.mark.parametrize("finish", ["commit", "rollback"])
@pytest.mark.parametrize("warm", [False, True])
def test_resident_public_snapshot_with_active_original_token(finish, warm):
    fast, reference = runtime_pair()
    core, oracle = (runtime.runtimes["hbm"].core for runtime in (fast, reference))
    if warm:
        for candidate in (core, oracle):
            candidate.submit(AccessRequest("warm", "read", 0, 4096))
    before = reference.snapshot()
    token = core.begin_transaction()
    request = AccessRequest("prefix", "write", 65536, 4096)
    assert core.submit(request) == oracle.submit(request)
    middle = reference.snapshot()
    assert token.numeric is not None
    saved = fast.snapshot()
    assert token.full is not None
    core.submit(AccessRequest("suffix", "read", 131072, 4096))
    fast.restore(saved)
    assert core._hbm_transactions[-1] is token
    getattr(core, finish + "_transaction")(token)
    expected = before if finish == "rollback" else middle
    # A core-only token protects controller state; runtime queues are outside it.
    actual = fast.snapshot()["runtime_state"]["hbm"]["core"]
    expected = expected["runtime_state"]["hbm"]["core"]
    for key in ("_banks", "_hbm_control_state"):
        assert actual[key] == expected[key]


@pytest.mark.parametrize("fallback", [False, True])
def test_resident_nested_runtime_rollback_and_retry(fallback):
    fast, reference = runtime_pair()
    core, oracle = (runtime.runtimes["hbm"].core for runtime in (fast, reference))
    initial = reference.snapshot()
    outer = fast.begin_transaction()
    prefix = AccessRequest("prefix", "write", 65536, 4096)
    assert core.submit(prefix) == oracle.submit(prefix)
    inner = fast.begin_transaction()
    suffix = AccessRequest("suffix", "read", 131072, 8192)
    core.submit(suffix)
    if fallback:
        core._hbm_before_reference()
    fast.restore(inner)
    assert core.submit(suffix) == oracle.submit(suffix)
    fast.restore(outer)
    assert visible(fast.snapshot()) == visible(initial)


def test_public_dictionary_alias_mutation_disables_residency():
    fast, reference = runtime_pair()
    core, oracle = (runtime.runtimes["hbm"].core for runtime in (fast, reference))
    warm = AccessRequest("warm", "read", 0, 4096)
    assert core.submit(warm) == oracle.submit(warm)
    aliases = [candidate._banks for candidate in (core, oracle)]
    request = AccessRequest("next", "write", 0, 4096)
    assert core.submit(request) == oracle.submit(request)
    for banks in aliases:
        for bank in banks.values():
            bank.ready_ns += 1234.
    assert core.submit(warm) == oracle.submit(warm)
    assert core._hbm_numeric_state is None
    assert visible(fast.snapshot()) == visible(reference.snapshot())


def test_external_timeline_reservations_are_loaded_before_resident_submit():
    fast, reference = runtime_pair()
    core, oracle = (runtime.runtimes["hbm"].core for runtime in (fast, reference))
    request = AccessRequest("warm", "read", 0, 4096)
    assert core.submit(request) == oracle.submit(request)
    for runtime in (fast, reference):
        rid = next(rid for rid in runtime.timeline.ready_ns if "data" in rid)
        runtime.timeline.ready_ns[rid] += 10000.
        runtime.timeline.lane_available[rid][0] += 10000.
    assert core.submit(request) == oracle.submit(request)
    assert visible(fast.snapshot()) == visible(reference.snapshot())


def test_b200_resident_full_banks_and_row_deadlines():
    from test_hbm_column_pack_equivalence import b200_memory

    config = b200_memory()
    fast = PhysicalRuntimeContext(config, "hbm", capture_details=False)
    reference = PhysicalRuntimeContext(config, "hbm", capture_details=False)
    core, oracle = (runtime.runtimes["hbm"].core for runtime in (fast, reference))
    oracle._hbm_numeric_enabled = False
    row_span = config.lane_count * config.bank_count * config.row_bytes
    requests = [
        AccessRequest("all-banks", "read", 0, config.lane_count * config.bank_count * config.burst_bytes),
        AccessRequest("kv-append", "write", 65539, 8192),
        AccessRequest("new-weight-row", "read", row_span + 31, 16384),
        AccessRequest("after-deadline", "read", 0, 4096, 40000.),
    ]
    for request in requests:
        transaction = fast.begin_transaction()
        assert core.submit(request) == oracle.submit(request)
        fast.commit_transaction(transaction)
        assert core._hbm_numeric_state is not None
    assert visible(fast.snapshot()) == visible(reference.snapshot())


def test_resident_numeric_exception_restores_uncommitted_arrays(monkeypatch):
    from heterollm_sim._compiled_hbm import _HBMNumericState

    fast, reference = runtime_pair()
    core, oracle = (runtime.runtimes["hbm"].core for runtime in (fast, reference))
    warm = AccessRequest("warm", "read", 0, 4096)
    assert core.submit(warm) == oracle.submit(warm)
    original = _HBMNumericState.execute

    def fail_after_execute(packed, request):
        original(packed, request)
        raise RuntimeError("injected numeric executor failure")

    with monkeypatch.context() as patch:
        patch.setattr(_HBMNumericState, "execute", fail_after_execute)
        with pytest.raises(RuntimeError, match="injected"):
            core.submit(AccessRequest("failed", "write", 65536, 8192))
    assert core.submit(warm) == oracle.submit(warm)
    assert visible(fast.snapshot()) == visible(reference.snapshot())


def test_resident_batch_failure_keeps_earlier_accepted_requests(monkeypatch):
    from heterollm_sim._compiled_hbm import _HBMNumericState

    fast, reference = runtime_pair()
    core, oracle = (runtime.runtimes["hbm"].core for runtime in (fast, reference))
    warm = AccessRequest("warm", "read", 0, 4096)
    assert core.submit(warm) == oracle.submit(warm)
    original = _HBMNumericState.execute
    first = AccessRequest("accepted", "write", 65536, 4096)
    second = AccessRequest("failed", "write", 131072, 8192)

    def fail_second(packed, request):
        result = original(packed, request)
        if request.request_id == "failed":
            raise RuntimeError("injected batch failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(_HBMNumericState, "execute", fail_second)
        with pytest.raises(RuntimeError, match="injected"):
            core.submit_many((first, second))
    oracle.submit(first)
    assert core.submit(warm) == oracle.submit(warm)
    assert visible(fast.snapshot()) == visible(reference.snapshot())


@pytest.mark.parametrize("mapping", ["bank_interleaved", "row_local"])
def test_channel_refresh_reads_external_changes_for_wrapped_and_scattered_batches(mapping):
    from dataclasses import replace
    from test_hbm_column_pack_equivalence import b200_memory

    config = replace(b200_memory(), address_mapping=mapping,
                     interleave_bytes=128 if mapping == "bank_interleaved" else None)
    fast = PhysicalRuntimeContext(config, "hbm", capture_details=False)
    reference = PhysicalRuntimeContext(config, "hbm", capture_details=False)
    core, oracle = (runtime.runtimes["hbm"].core for runtime in (fast, reference))
    oracle._hbm_numeric_enabled = False
    seed = AccessRequest("seed", "read", 0, config.lane_count * config.bank_count * config.row_bytes)
    assert core.submit(seed) == oracle.submit(seed)
    ids = core._hbm_numeric_state.ids
    channels = config.lane_count // 2
    for runtime in (fast, reference):
        for index in (0, channels - 1, channels, 2 * channels - 1,
                      2 * channels, 4 * channels - 1):
            rid = ids[index]
            runtime.timeline.lane_available[rid][0] += 1000.125
            runtime.timeline.ready_ns[rid] = runtime.timeline.lane_available[rid][0]
            runtime.timeline.busy_ns[rid] += .125
            runtime.timeline.last_intervals[rid] = (900., 1000.125)
            if index >= 2 * channels:
                runtime.timeline.bytes_moved[rid] += 32
                runtime.timeline.directions[rid] = "write"
    span = config.interleave_bytes or config.row_bytes * config.bank_count
    unrelated = AccessRequest("middle-channel", "read", 20 * span, 32)
    def physical(result):
        return replace(result, counters={key: value for key, value in result.counters.items()
                                         if key != "compiled_hbm_float64"})
    assert physical(core.submit(unrelated)) == physical(oracle.submit(unrelated))
    requests = (AccessRequest("zero", "read", 0, 32),
                AccessRequest("wrap", "read", config.lane_count * span - 17, 65),
                AccessRequest("another", "write", 7 * span, 32))
    assert tuple(map(physical, core.submit_many(requests))) == tuple(map(physical, oracle.submit_many(requests)))
    assert visible(fast.snapshot()) == visible(reference.snapshot())
