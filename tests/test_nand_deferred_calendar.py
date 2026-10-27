"""Native READ batching preserves ordered requests, publication and rollback.

Normal cases do not wrap native writers; their oracle is ordered individual
submits. Callbacks are installed only when qualification must decline.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest

ROOT = next(path for path in Path(__file__).resolve().parents
            if (path / "tests/test_generic_nand_resident.py").is_file())
sys.path.insert(0, str(ROOT / "tests"))

from heterollm_sim import _nand_acceleration as acceleration
from heterollm_sim._nand_execution import _geometry_plan
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_types import AccessRequest, NandConfig, Operation, TransactionResult
from test_generic_nand_resident import logical
from test_nand_generic_buffered_acceleration import PAGE, config
from test_token_owned_history import bits


def pair(settings=None):
    settings = settings or config(host_granularity_bytes=256, max_outstanding_requests=2)
    runtimes = [PhysicalRuntimeContext(settings, "flash", capture_details=False) for _ in range(2)]
    for runtime in runtimes:
        core = runtime.runtimes["flash"].core
        core.submit(AccessRequest("warm", "read", 0, 128 * PAGE, .125))
        assert core._nand_numeric_state is not None
    return runtimes


def cores(runtimes):
    return tuple(runtime.runtimes["flash"].core for runtime in runtimes)


def requests(base=256, count=10):
    # Two-page partial reads run the resident generic engine; varying banks
    # exercise union versus last-request-only publication.
    return tuple(AccessRequest(f"r{index}", "read", (base + index * 3) * PAGE + 17,
                               2 * PAGE - 33, .125) for index in range(count))


def calendar_args(core):
    cfg = core.config
    base = cfg.channels * cfg.targets_per_channel * cfg.dies_per_target * cfg.luns_per_die
    banks = base * cfg.planes_per_lun
    units = banks if cfg.planes_independent else base
    plan = _geometry_plan(core, Operation.READ, banks, units, base // cfg.channels)
    _, _, arrays, resources, channels, hosts, names, *_ = plan
    valid, bus_mask = acceleration._buffered_layout_contract(
        arrays, resources, channels, hosts, names, tuple(core.timeline.resource_aliases.items()))
    assert valid
    return names, arrays, resources, bus_mask, True


def eligible(core):
    return acceleration._deferred_calendar_eligible(core, *calendar_args(core))


def compare(runtimes, batch):
    fast, reference = cores(runtimes)
    actual = fast.submit_many(batch)
    expected = tuple(reference.submit(request) for request in batch)
    assert bits(actual) == bits(expected)
    assert bits(logical(fast)) == bits(logical(reference))
    assert "_nand_request_metric_orders" not in fast.__dict__
    return actual, expected


def assert_error_prefix(runtimes, batch, expected_type, message=None):
    errors = []
    for index, core in enumerate(cores(runtimes)):
        with pytest.raises(expected_type) as caught:
            if index == 0:
                core.submit_many(batch)
            else:
                for request in batch:
                    core.submit(request)
        errors.append((type(caught.value), str(caught.value)))
        assert "_nand_request_metric_orders" not in core.__dict__
    assert errors[0] == errors[1]
    if message is not None:
        assert message in errors[0][1]
    assert bits(logical(cores(runtimes)[0])) == bits(logical(cores(runtimes)[1]))


@pytest.mark.parametrize("depth", [1, 3])
def test_native_batch_matches_each_request_admission_and_public_state(depth):
    runtimes = pair(config(host_granularity_bytes=256, max_outstanding_requests=depth))
    batch = requests()
    for core in cores(runtimes):
        assert acceleration._deferred_batch_qualified(core, batch)
        assert eligible(core)
    actual, _ = compare(runtimes, batch)
    assert all(result.counters["native_fullpage_pages"] == 2 for result in actual)
    assert any(result.counters["queue_wait_ns"] > 0 for result in actual)
    assert len({frozenset(result.counters["resource_last_intervals"]) for result in actual}) > 1


def test_union_flush_keeps_original_lane_lists_last_tuple_aliases_and_foreign_state():
    runtimes = pair()
    held_lanes, held_last = [], []
    for runtime in runtimes:
        row = runtime.timeline
        row.ready_ns["foreign-owner"] = 731.125
        row.busy_ns["foreign-owner"] = .375
        row.bytes_moved["foreign-owner"] = 2**70 + 3
        row.last_intervals["foreign-owner"] = (71.125, 731.125)
        row.lane_available["foreign-owner"] = [731.125]
        held_lanes.append(dict(row.lane_available))
        held_last.append(dict(row.last_intervals))
    actual, expected = compare(runtimes, requests())
    for runtime, output, lanes, previous_last in zip(runtimes, (actual, expected), held_lanes, held_last):
        row = runtime.timeline
        assert all(row.lane_available[rid] is value for rid, value in lanes.items())
        latest = {}
        for result in output:
            latest.update(result.counters["resource_last_intervals"])
        for rid, interval in latest.items():
            assert row.last_intervals[rid] is interval
        for rid, interval in previous_last.items():
            if rid not in latest:
                assert row.last_intervals[rid] is interval
        assert row.ready_ns["foreign-owner"] == 731.125
        assert row.bytes_moved["foreign-owner"] == 2**70 + 3


def test_same_resource_many_times_keeps_final_array_and_lane_ready():
    settings = config(channels=1, targets_per_channel=1, dies_per_target=1,
                      planes_per_lun=1, max_outstanding_requests=1)
    runtimes = pair(settings)
    actual, expected = compare(runtimes, requests(count=12))
    for core, output in zip(cores(runtimes), (actual, expected)):
        names, arrays, resources, _, _ = calendar_args(core)
        assert arrays == resources
        for rid in arrays:
            assert core.timeline.ready_ns[rid] == core.timeline.lane_available[rid][0]
        latest = {}
        for result in output:
            latest.update(result.counters["resource_last_intervals"])
        assert all(core.timeline.last_intervals[rid] is interval for rid, interval in latest.items())


def test_foreign_lane_directions_do_not_disable_own_calendar_or_change_aliases():
    runtimes = pair()
    held = []
    for runtime in runtimes:
        row = runtime.timeline
        foreign_lanes = [73.125, 81.375]
        foreign_directions = ["read", "write"]
        row.ready_ns["foreign-hbm"] = 73.125
        row.busy_ns["foreign-hbm"] = 8.25
        row.bytes_moved["foreign-hbm"] = 512
        row.last_intervals["foreign-hbm"] = (70.125, 73.125)
        row.lane_available["foreign-hbm"] = foreign_lanes
        row.lane_directions["foreign-hbm"] = foreign_directions
        held.append((foreign_lanes, foreign_directions, row.last_intervals["foreign-hbm"]))
        core = runtime.runtimes["flash"].core
        assert eligible(core)
        assert acceleration._deferred_batch_qualified(core, requests())
    compare(runtimes, requests())
    for runtime, (lanes, directions, interval) in zip(runtimes, held):
        row = runtime.timeline
        assert row.lane_available["foreign-hbm"] is lanes
        assert row.lane_directions["foreign-hbm"] is directions
        assert row.last_intervals["foreign-hbm"] is interval
        assert lanes == [73.125, 81.375]
        assert directions == ["read", "write"]
        assert row.ready_ns["foreign-hbm"] == 73.125
        assert row.busy_ns["foreign-hbm"] == 8.25
        assert row.bytes_moved["foreign-hbm"] == 512


def test_out_of_range_suffix_flushes_only_successful_prefix():
    runtimes = pair()
    batch = requests(count=8) + (AccessRequest("invalid", "read",
        cores(runtimes)[0].config.effective_capacity_bytes, 2 * PAGE),)
    assert_error_prefix(runtimes, batch, ValueError)


def test_real_finite_overflow_does_not_publish_dirty_failed_calendar():
    settings = config(channels=1, targets_per_channel=1, dies_per_target=1,
                      planes_per_lun=1, max_outstanding_requests=32)
    runtimes = pair(settings)
    for core in cores(runtimes):
        # Preserve a warmed resident layout. Each fresh two-page request adds
        # 2e307 to its one array: eight succeed; the ninth becomes non-finite.
        core.config = replace(core.config, page_read_ns=1e307)
    batch = tuple(AccessRequest(f"overflow-{index}", "read", (256 + index * 2) * PAGE,
                                2 * PAGE, .125) for index in range(10))
    for core in cores(runtimes):
        assert acceleration._deferred_batch_qualified(core, batch)
        assert eligible(core)
    assert_error_prefix(runtimes, batch, ValueError, "finite timing range")


@pytest.mark.parametrize("inner_finish", ["commit_transaction", "restore"])
def test_nested_outer_undo_retains_original_public_preimages(inner_finish):
    runtimes = pair()
    assert all(eligible(core) for core in cores(runtimes))
    before = [bits(logical(core)) for core in cores(runtimes)]
    outer = [runtime.begin_transaction(owners=("flash",), _owned_timeline=True) for runtime in runtimes]
    compare(runtimes, requests())
    inner = [runtime.begin_transaction(owners=("flash",), _owned_timeline=True) for runtime in runtimes]
    compare(runtimes, requests(base=512))
    for runtime, token in zip(runtimes, inner):
        getattr(runtime, inner_finish)(token)
    compare(runtimes, requests(base=768))
    for runtime, token, saved in zip(runtimes, outer, before):
        runtime.restore(token)
        assert bits(logical(runtime.runtimes["flash"].core)) == saved


def test_single_page_scalar_handoff_reads_successful_pending_calendars():
    runtimes = pair()
    assert all(eligible(core) for core in cores(runtimes))
    batch = list(requests(count=10))
    batch[4] = AccessRequest("scalar-page", "read", 911 * PAGE + 17, 53, .125)
    compare(runtimes, tuple(batch))


@pytest.mark.parametrize("change", ["same_root", "missing_ready", "missing_busy",
                                    "missing_bytes", "shared_lanes", "multilane"])
def test_calendar_qualification_rejects_unsafe_roots_keys_and_lanes(change):
    core = cores(pair())[0]
    names, arrays, resources, bus, _ = calendar_args(core)
    row = core.timeline
    bus_id = next(name for name, flag in zip(names, bus) if flag)
    if change == "same_root":
        # Qualification only: the pre-existing shared_pack path has a separate
        # alias edge case; this patch must not silently redefine that baseline.
        row.busy_ns = row.ready_ns
    elif change == "missing_ready":
        row.ready_ns.pop(names[0])
    elif change == "missing_busy":
        row.busy_ns.pop(names[0])
    elif change == "missing_bytes":
        row.bytes_moved.pop(bus_id)
    elif change == "shared_lanes":
        row.lane_available[names[1]] = row.lane_available[names[0]]
    else:
        row.lane_available[bus_id].append(17.125)
    assert not eligible(core)


def test_new_calendar_key_uses_existing_publication_order():
    runtimes = pair()
    for core in cores(runtimes):
        names, *_ = calendar_args(core)
        core.timeline.busy_ns.pop(names[0])
        assert not eligible(core)
    compare(runtimes, requests())


@pytest.mark.parametrize("where", ["request", "config", "calendar"])
def test_numeric_subclasses_do_not_enter_callback_free_deferred_scope(where):
    class AuthoredFloat(float):
        pass

    core = cores(pair())[0]
    batch = requests()
    if where == "request":
        batch = (replace(batch[0], arrival_ns=AuthoredFloat(.125)),) + batch[1:]
        assert not acceleration._deferred_batch_qualified(core, batch)
    elif where == "config":
        core.config = replace(core.config, front_ns=AuthoredFloat(.125))
        assert not acceleration._deferred_batch_qualified(core, batch)
    else:
        names, *_ = calendar_args(core)
        core.timeline.busy_ns[names[0]] = AuthoredFloat(core.timeline.busy_ns[names[0]])
        assert not eligible(core)


def test_parallel_prefix_alias_rejects_union_controller_ready_overwrite():
    settings = config(channels=1, targets_per_channel=1, dies_per_target=3, planes_per_lun=1,
                      parallel_units=1, metadata={"array_resource_prefix": "same-array",
                                                 "parallel_resource_prefix": "same-array"})
    runtimes = pair(settings)
    for core in cores(runtimes):
        assert not eligible(core)
    # First read misses units 0/1; next read hits unit 1 and misses unit 2.
    # Replaying the union of controller frontiers at batch end would let
    # unit 0's older frontier overwrite the latest shared physical array.
    batch = (AccessRequest("units-0-1", "read", 0, 2 * PAGE),) + tuple(
        AccessRequest(f"units-1-2-{index}", "read", PAGE, 2 * PAGE) for index in range(7))
    compare(runtimes, batch)
    core = cores(runtimes)[0]
    _, arrays, resources, _, _ = calendar_args(core)
    assert resources[0] == resources[1] == resources[2]
    numeric_ready = core._nand_numeric_state.arrays[0]
    assert core.timeline.ready_ns[resources[0]] == float(numeric_ready[2])
    assert float(numeric_ready[2]) > float(numeric_ready[0])


def test_config_callback_observes_complete_previous_publication(monkeypatch):
    runtimes = pair()
    active = [None]
    observations = [[], []]
    original = NandConfig.host_bandwidth_for

    def observed(settings, operation):
        if active[0] is not None:
            index = active[0]
            core = cores(runtimes)[index]
            if settings is core.config:
                observations[index].append(bits(dict(core.timeline.ready_ns)))
        return original(settings, operation)

    monkeypatch.setattr(NandConfig, "host_bandwidth_for", observed)
    batch = requests()
    for core in cores(runtimes):
        assert not acceleration._deferred_batch_qualified(core, batch)
    active[0] = 0
    actual = cores(runtimes)[0].submit_many(batch)
    active[0] = 1
    expected = tuple(cores(runtimes)[1].submit(request) for request in batch)
    active[0] = None
    assert observations[0] and observations[0] == observations[1]
    assert bits(actual) == bits(expected)
    assert bits(logical(cores(runtimes)[0])) == bits(logical(cores(runtimes)[1]))


def test_constructor_failure_after_publication_keeps_original_successful_prefix(monkeypatch):
    runtimes = pair()
    original = TransactionResult.__init__

    def failing(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.request_id == "r5":
            raise RuntimeError("after request calendar publication")

    monkeypatch.setattr(TransactionResult, "__init__", failing)
    batch = requests()
    for core in cores(runtimes):
        assert not acceleration._deferred_batch_qualified(core, batch)
    # This is deliberately an unqualified callback path, not a fault injected
    # into a native writer while pretending its native proof still holds.
    assert_error_prefix(runtimes, batch, RuntimeError, "after request calendar publication")



def test_deferred_array_ready_misses_hits_and_late_first_touches_keep_aliases():
    runtimes = pair()
    batch = tuple(AccessRequest(f"ready-{index}", "read",
        (256 + (index // 2) * 4) * PAGE, 2 * PAGE, .125) for index in range(12))
    held = []
    for core in cores(runtimes):
        assert eligible(core)
        held.append((dict(core.timeline.ready_ns), dict(core.timeline.last_intervals)))
    actual, expected = compare(runtimes, batch)
    assert all(result.counters["read_cache_misses"] == 2 for result in actual[::2])
    assert all(result.counters["read_cache_hits"] == 2 for result in actual[1::2])
    signatures = []
    for core, output, (previous_ready, previous_last) in zip(
            cores(runtimes), (actual, expected), held):
        _, arrays, *_ = calendar_args(core)
        latest = {}
        for result in output:
            latest.update(result.counters["resource_last_intervals"])
        changed_arrays = set(arrays).intersection(latest)
        assert len(changed_arrays) > 2
        assert len(changed_arrays) < len(arrays)
        assert all(core.timeline.ready_ns[rid] is previous_ready[rid]
                   for rid in arrays if rid not in changed_arrays)
        assert all(core.timeline.ready_ns[rid] is not previous_ready[rid]
                   for rid in changed_arrays)
        assert all(core.timeline.last_intervals[rid] is interval
                   for rid, interval in latest.items())
        assert all(core.timeline.last_intervals[rid] is interval
                   for rid, interval in previous_last.items() if rid not in latest)
        signatures.append(tuple((rid,
            core.timeline.ready_ns[rid] is previous_ready[rid],
            core.timeline.ready_ns[rid] is core.timeline.lane_available[rid][0],
            core.timeline.ready_ns[rid] is core.timeline.last_intervals[rid][0],
            core.timeline.ready_ns[rid] is core.timeline.last_intervals[rid][1]) for rid in arrays))
    assert signatures[0] == signatures[1]


def test_deferred_array_ready_hit_only_batch_keeps_all_array_objects():
    runtimes = pair()
    batch = tuple(AccessRequest(f"cached-{index}", "read", 126 * PAGE,
                               2 * PAGE, .125) for index in range(8))
    held = []
    for core in cores(runtimes):
        assert eligible(core)
        _, arrays, *_ = calendar_args(core)
        held.append({rid: (core.timeline.ready_ns[rid], core.timeline.last_intervals[rid])
                     for rid in arrays})
    actual, _ = compare(runtimes, batch)
    assert all(result.counters["read_cache_hits"] == 2 for result in actual)
    for core, previous in zip(cores(runtimes), held):
        assert all(core.timeline.ready_ns[rid] is ready
                   and core.timeline.last_intervals[rid] is interval
                   for rid, (ready, interval) in previous.items())


@pytest.mark.parametrize("field", ["_array_ready", "_buffer_ready"])
def test_authored_frontier_dict_keeps_original_controller_path(field):
    settings = config(channels=1, targets_per_channel=1, dies_per_target=1,
                      planes_per_lun=1, max_outstanding_requests=1)
    runtimes = pair(settings)
    calls = [[], []]

    def wrapped(values, timeline, observations):
        class AuthoredFrontier(dict):
            def __contains__(self, key):
                observations.append((bits(dict(timeline.ready_ns)), bits(dict(timeline.busy_ns))))
                return super().__contains__(key)
        return AuthoredFrontier(values)

    for index, core in enumerate(cores(runtimes)):
        core.__dict__[field] = wrapped(core.__dict__[field], core.timeline, calls[index])
        assert not eligible(core)
    # Qualification itself must not enter authored membership. Discard any
    # setup prefix explicitly, then compare every complete public observation.
    before = [len(items) for items in calls]
    compare(runtimes, requests(count=10))
    actual_calls, expected_calls = (items[count:] for items, count in zip(calls, before))
    assert actual_calls == expected_calls
    if field == "_array_ready":
        assert len(actual_calls) >= 10


@pytest.mark.parametrize("alias", ["array_ready", "buffer_ready", "same_frontiers"])
def test_frontier_root_alias_keeps_original_controller_path(alias):
    runtimes = pair()
    for core in cores(runtimes):
        if alias == "array_ready":
            core.__dict__["_array_ready"] = core.timeline.ready_ns
        elif alias == "buffer_ready":
            core.__dict__["_buffer_ready"] = core.timeline.ready_ns
        else:
            core.__dict__["_buffer_ready"] = core.__dict__["_array_ready"]
        assert not eligible(core)
    compare(runtimes, requests())



def test_authored_cache_membership_observes_complete_public_calendars():
    runtimes = pair()
    observations = [[], []]

    def wrapped(values, timeline, output):
        class AuthoredCache(dict):
            def __contains__(self, key):
                output.append((bits(dict(timeline.ready_ns)), bits(dict(timeline.busy_ns))))
                return super().__contains__(key)
        return AuthoredCache(values)

    for index, core in enumerate(cores(runtimes)):
        core.__dict__["_read_cache"] = wrapped(core.__dict__["_read_cache"], core.timeline, observations[index])
        assert not eligible(core)
    before = [len(output) for output in observations]
    compare(runtimes, requests())
    actual, expected = (output[start:] for output, start in zip(observations, before))
    assert actual and actual == expected


@pytest.mark.parametrize("change", ["tuple_subclass", "part_subclass", "root_alias"])
def test_unknown_cache_keys_or_root_do_not_defer(change):
    class Bank(tuple):
        pass
    class Part(int):
        pass
    core = cores(pair())[0]
    cache = core.__dict__["_read_cache"]
    if change == "root_alias":
        core.__dict__["_read_cache"] = core.timeline.lane_directions
    else:
        key = next(iter(cache))
        row = cache.pop(key)
        authored = Bank(key) if change == "tuple_subclass" else (Part(key[0]), *key[1:])
        cache[authored] = row
    assert not eligible(core)


def test_authored_native_cache_journal_observes_complete_public_calendars():
    settings = config(channels=1, targets_per_channel=1, dies_per_target=1,
                      planes_per_lun=256, planes_independent=False)
    runtimes = pair(settings)
    transactions = [runtime.begin_transaction(owners=("flash",), _owned_timeline=True)
                    for runtime in runtimes]
    observations = [[], []]

    def wrapped(values, timeline, output):
        class AuthoredJournal(dict):
            def __contains__(self, key):
                output.append((bits(dict(timeline.ready_ns)), bits(dict(timeline.busy_ns))))
                return super().__contains__(key)
        return AuthoredJournal(values)

    batch = requests(base=128, count=8)
    for index, core in enumerate(cores(runtimes)):
        token = core._nand_transactions[-1]
        token.cache_banks = wrapped(token.cache_banks, core.timeline, observations[index])
        assert not acceleration._deferred_batch_qualified(core, batch)
    before = [len(output) for output in observations]
    compare(runtimes, batch)
    actual, expected = (output[start:] for output, start in zip(observations, before))
    assert actual and actual == expected
    for runtime, token in zip(runtimes, transactions):
        runtime.commit_transaction(token)


@pytest.mark.parametrize("change", ["accounting", "roots_key", "entries_key", "saved_key", "object_key"])
def test_authored_owned_journal_values_cannot_enter_private_scope(change):
    class Key(str):
        pass
    class Identity(int):
        pass
    class Accounting:
        def __bool__(self):
            raise AssertionError("qualification must not invoke authored truth conversion")
    runtime = pair()[0]
    core = runtime.runtimes["flash"].core
    runtime.begin_transaction(owners=("flash",), _owned_timeline=True)
    token = core.timeline._owned_transactions[-1]
    if change == "accounting":
        token.accounting = Accounting()
    elif change in ("roots_key", "entries_key"):
        field = "roots" if change == "roots_key" else "entries"
        values = getattr(token, field)
        setattr(token, field, {Key(name) if name == "ready_ns" else name: item
                              for name, item in values.items()})
    elif change == "saved_key":
        token.entries["ready_ns"][Key(next(iter(core.timeline.ready_ns)))] = 0.
    else:
        token.objects[Identity(1)] = None
    assert not acceleration._deferred_batch_qualified(core, requests())


@pytest.mark.parametrize("change", ["float32", "array_alias", "view", "dirty_subclass",
                                    "arrays_subclass", "layout_subclass", "contract_subclass"])
def test_non_native_numeric_storage_and_descriptors_do_not_defer(change):
    import numpy as np
    class Array(np.ndarray):
        pass
    class Items(tuple):
        pass
    core = cores(pair())[0]
    packed = core._nand_numeric_state
    if change == "float32":
        packed.arrays = (packed.arrays[0].astype(np.float32), *packed.arrays[1:])
    elif change == "array_alias":
        packed.arrays = (packed.arrays[0], packed.arrays[0], *packed.arrays[2:])
    elif change == "view":
        packed.arrays = (packed.arrays[0].view(), *packed.arrays[1:])
    elif change == "dirty_subclass":
        packed.dirty_units = packed.dirty_units.view(Array)
    elif change == "arrays_subclass":
        packed.arrays = Items(packed.arrays)
    elif change == "layout_subclass":
        packed.layout = Items(packed.layout)
    else:
        packed.scalar_resource_contract = Items(packed.scalar_resource_contract)
    assert not acceleration._deferred_batch_qualified(core, requests())


def test_native_equal_copied_numeric_layout_still_qualifies():
    core = cores(pair())[0]
    packed = core._nand_numeric_state
    bank_keys, prefixes, names, pages = packed.layout
    packed.layout = (tuple(tuple(list(key)) for key in bank_keys),
                     tuple(tuple(list(key)) for key in prefixes), tuple(list(names)), pages)
    packed.scalar_resource_contract = tuple(list(packed.scalar_resource_contract))
    assert acceleration._deferred_batch_qualified(core, requests())
    assert eligible(core)
