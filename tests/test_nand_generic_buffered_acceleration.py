"""Strict generic page batching keeps the original double-buffer calendars."""
from copy import deepcopy
from dataclasses import replace

import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest, NandConfig
from heterollm_sim.nand_core import NandCore


PAGE = 4096


def config(**changes):
    values = dict(channels=2, targets_per_channel=2, dies_per_target=2,
                  planes_per_lun=3, planes_independent=True, page_bytes=PAGE,
                  read_cache_pages_per_bank=2, host_full_duplex=True,
                  host_bandwidth_gb_s=17., host_read_bandwidth_gb_s=17.,
                  host_write_bandwidth_gb_s=7., host_granularity_bytes=PAGE,
                  internal_bandwidth_gb_s=3.1, page_read_ns=91.5,
                  page_program_ns=351.1, front_ns=.1, max_expanded_segments=300)
    values.update(changes)
    return NandConfig(**values)


def cores(settings, timeline=None):
    fast = NandCore(settings, deepcopy(timeline), capture_details=False)
    scalar = NandCore(settings, deepcopy(timeline), capture_details=False)
    scalar._execute_accepted = scalar._execute_detailed_accepted
    return fast, scalar


def assert_state(fast, scalar):
    # No approximate float comparison: ordered reservation arithmetic is
    # unchanged, including previous busy totals and multi-lane histories.
    assert fast.timeline == scalar.timeline
    assert fast.timeline._touched == scalar.timeline._touched
    assert fast._array_ready == scalar._array_ready
    assert fast._buffer_ready == scalar._buffer_ready
    assert fast._read_cache == scalar._read_cache
    for name in ("_cache_use", "_acceptance_ns", "_host_acceptance_ns", "_host_inflight",
                 "_inflight", "_inflight_operations", "_next_program_page", "_block_wear_state"):
        assert getattr(fast, name) == getattr(scalar, name)


def submit_pair(pair, request, *, native=None):
    fast, scalar = pair
    actual, expected = fast.submit(request), scalar.submit(request)
    counters = dict(actual.counters)
    executed = counters.pop("native_fullpage_pages", None)
    if native is not None:
        assert bool(executed) == native
    assert replace(actual, counters=counters) == expected
    assert_state(fast, scalar)
    return actual


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("full_duplex", [False, True])
@pytest.mark.parametrize("host_channels", [1, 2])
def test_mixed_bulk_partial_hits_erase_and_direction_calendars(independent, full_duplex, host_channels):
    pair = cores(config(planes_independent=independent, host_full_duplex=full_duplex,
                        host_channels=host_channels, internal_transfer_bytes=3072))
    operations = [
        ("read", 0, 256 * PAGE, 0.),
        ("read", 240 * PAGE, 256 * PAGE, 10.),  # real cache hits in the first stripe
        ("write", 7 * PAGE, 256 * PAGE, .3),
        ("read", 0, 256 * PAGE, .4),
        ("write", 29, 128 * PAGE + 13, 1.),  # scalar RMW boundary semantics
        ("read", 35, 128 * PAGE + 31, 1.),
        ("erase", 0, PAGE, 2.),
        ("read", 0, 129 * PAGE, 2.),
        ("read", 0, PAGE, 2.),
    ]
    for index, (operation, address, count, arrival) in enumerate(operations):
        native = count >= 128 * PAGE and not address % PAGE and not count % PAGE and operation != "erase"
        submit_pair(pair, AccessRequest(str(index), operation, address, count, arrival), native=native)
    timeline = pair[0].timeline
    if full_duplex:
        assert any(name.endswith(":read") for name in timeline.bytes_moved)
        assert any(name.endswith(":write") for name in timeline.bytes_moved)


def test_two_slot_overwrite_waits_for_reader_after_write_invalidation():
    settings = config(channels=1, targets_per_channel=1, dies_per_target=1, planes_per_lun=1,
                      host_bandwidth_gb_s=1., host_read_bandwidth_gb_s=1., host_write_bandwidth_gb_s=.5)
    pair = cores(settings)
    # Two fills, one hit, then an invalidation retain live reader lifetimes.
    for index, (operation, page) in enumerate((("read", 0), ("read", 1), ("read", 0), ("write", 0))):
        submit_pair(pair, AccessRequest(str(index), operation, page * PAGE, PAGE))
    old_readers = [slot.reader_completion_ns for slot in pair[0]._read_cache[(0, 0, 0, 0, 0)]]
    for core in pair:
        core.timeline.lane_available["nand:host:read"] = [0., 0.]
    result = submit_pair(pair, AccessRequest("replace", "read", 2 * PAGE, 128 * PAGE), native=True)
    assert result.counters["read_cache_misses"] == 128
    assert min(slot.ready_ns for slot in pair[0]._read_cache[(0, 0, 0, 0, 0)]) >= min(old_readers)


@pytest.mark.parametrize("parallel", [None, 2])
def test_existing_multilane_and_shared_array_execution_slots(parallel):
    timeline = ResourceTimeline(lane_available={
        "nand:array:0": [0., 104.], "nand:parallel:0": [0., 1000.],
        "nand:channel:0": [20., 40.], "nand:host:read": [0., 70.],
        "nand:host:write": [800., 3.],
    }, directions={"nand:channel:0": "write"},
       lane_directions={"nand:channel:0": ["read", "write"]})
    pair = cores(config(parallel_units=parallel), timeline)
    for operation in ("read", "write", "read"):
        submit_pair(pair, AccessRequest(operation, operation, 0, 384 * PAGE), native=True)


def test_all_hits_do_not_create_unused_media_frontiers_or_change_counters():
    pair = cores(config(channels=64, planes_per_lun=1))
    submit_pair(pair, AccessRequest("seed", "read", 0, 128 * PAGE), native=True)
    result = submit_pair(pair, AccessRequest("all-hit", "read", 0, 128 * PAGE), native=True)
    assert result.counters["pages_read"] == result.transfer_bytes == 0
    assert result.counters["read_cache_hits"] == 128
    assert set(result.counters["resource_bytes"]) == {"nand:host:read"}


@pytest.mark.parametrize("finish", ["commit", "rollback"])
def test_runtime_nested_rollback_clone_and_scalar_handoff(finish):
    settings = config()
    runtimes = [PhysicalRuntimeContext(config=settings, physical_owner="flash", capture_details=False)
                for _ in range(2)]
    pair = [runtime.runtimes["flash"].core for runtime in runtimes]
    scalar_core = pair[1]
    # A plain test callable remains the same object across public snapshots;
    # deepcopy of an instance-bound method would instead copy its owner.
    pair[1]._execute_accepted = lambda request: scalar_core._execute_detailed_accepted(request)
    submit_pair(pair, AccessRequest("seed", "read", 0, 128 * PAGE), native=True)
    before = [runtime.snapshot() for runtime in runtimes]
    checkpoints = [runtime.begin_transaction() for runtime in runtimes]
    submit_pair(pair, AccessRequest("bulk-write", "write", 0, 256 * PAGE), native=True)
    inner = [core.begin_transaction() for core in pair]
    submit_pair(pair, AccessRequest("partial", "write", 11, PAGE + 13), native=False)
    submit_pair(pair, AccessRequest("bulk-read", "read", 0, 384 * PAGE), native=True)
    for core, token in zip(pair, inner):
        getattr(core, finish + "_transaction")(token)
    # A core token deliberately does not own the shared resource timeline;
    # full runtime rollback restores both controller and calendar state.
    for runtime, saved in zip(runtimes, checkpoints):
        runtime.restore(saved)
    assert_state(*pair)
    for runtime, saved in zip(runtimes, before):
        assert runtime.snapshot() == saved
    clone = deepcopy(runtimes[0])
    cloned_core = clone.runtimes["flash"].core
    submit_pair((cloned_core, pair[1]), AccessRequest("clone", "read", 30 * PAGE, 256 * PAGE), native=True)
    assert runtimes[0].snapshot() == before[0]


@pytest.mark.parametrize("change", [
    {"page_bytes": 1024}, {"read_cache_pages_per_bank": 1},
    {"host_granularity_bytes": 1536}, {"write_buffer_bytes": 1024 * 1024},
])
def test_nonmatching_semantics_retain_original_scalar_path(change):
    settings = config(**change)
    pair = cores(settings)
    submit_pair(pair, AccessRequest("fallback", "write", 0, 256 * settings.page_bytes), native=False)


def test_alias_to_shared_bus_retains_original_scalar_transitions():
    settings = config(metadata={"array_resource_prefix": "shared", "channel_resource_prefix": "shared",
                                "host_resource_id": "shared:0"})
    pair = cores(settings)
    submit_pair(pair, AccessRequest("aliases", "read", 0, 256 * PAGE), native=False)


def test_large_bank_geometry_keeps_tiny_spans_scalar_and_bulk_native():
    pair = cores(config(channels=7, dies_per_target=16, targets_per_channel=1,
                        planes_per_lun=25))
    submit_pair(pair, AccessRequest("tiny", "read", 0, 127 * PAGE), native=False)
    submit_pair(pair, AccessRequest("pd-chunk", "write", 17 * PAGE, 256 * PAGE), native=True)
    submit_pair(pair, AccessRequest("bulk", "read", 0, 4096 * PAGE), native=True)


def test_actual_112_channel_25_bank_pd_chunks_keep_exact_native_results_and_controller_state():
    settings = config(channels=112, targets_per_channel=1, dies_per_target=1,
        planes_per_lun=25, planes_independent=True, max_expanded_segments=100000)
    contexts = [PhysicalRuntimeContext(config=settings, physical_owner="dev", capture_details=False)
                for _ in range(2)]
    fast, scalar = [context.runtimes["dev"].core for context in contexts]
    scalar._execute_accepted = lambda request: scalar._execute_detailed_accepted(request)
    requests = [("write", 0, 1024*PAGE), ("write", 1024*PAGE, 256*PAGE),
        ("read", 1024*PAGE, 256*PAGE), ("write", 127*PAGE, 256*PAGE),
        ("read", 127*PAGE, 256*PAGE), ("write", 9, PAGE+17),
        ("write", 3001*PAGE, 256*PAGE), ("read", 3001*PAGE, 256*PAGE)]
    for index, (operation, address, count) in enumerate(requests):
        request = AccessRequest(str(index), operation, address, count, index*.1)
        actual, expected = fast.submit(request), scalar.submit(request)
        counters = dict(actual.counters)
        pages = counters.pop("native_fullpage_pages", None)
        if count == 256*PAGE:
            assert pages == 256
            assert fast._nand_numeric_state is not None
        assert replace(actual, counters=counters) == expected
    assert_state(fast, scalar)


def test_native_layout_reuse_rechecks_alias_values_and_mutable_resource_prefixes():
    from heterollm_sim._nand_acceleration import _buffered_layout_contract
    settings = config()
    contexts = [PhysicalRuntimeContext(config=settings, physical_owner="dev", capture_details=False)
                for _ in range(2)]
    fast, scalar = [context.runtimes["dev"].core for context in contexts]
    scalar._execute_accepted = lambda request: scalar._execute_detailed_accepted(request)
    before = _buffered_layout_contract.cache_info().hits
    for index in range(4):
        if index == 2:
            for core in (fast, scalar):
                core.timeline.resource_aliases[core.config.metadata["host_resource_id"] + ":write"] = "rerouted:write"
        if index == 3:
            for core in (fast, scalar):
                core.config.metadata["channel_resource_prefix"] = "rerouted:channel"
        request = AccessRequest(str(index), "write", index*256*PAGE, 256*PAGE, index*.125)
        actual, expected = fast.submit(request), scalar.submit(request)
        counters = dict(actual.counters)
        assert counters.pop("native_fullpage_pages") == 256
        assert replace(actual, counters=counters) == expected
    assert _buffered_layout_contract.cache_info().hits > before
    assert "rerouted:write" in fast.timeline.bytes_moved
    assert any(name.startswith("rerouted:channel:") for name in fast.timeline.ready_ns)
    assert_state(fast, scalar)


@pytest.mark.parametrize("extension", ["slot_alias", "slot_attribute", "empty_bank", "callback"])
def test_nonordinary_public_state_declines_before_native_mutation(extension):
    pair = cores(config())
    submit_pair(pair, AccessRequest("seed", "read", 0, PAGE), native=False)
    for core in pair:
        slots = core._read_cache[(0, 0, 0, 0, 0)]
        if extension == "slot_alias":
            slots[1] = slots[0]
        elif extension == "slot_attribute":
            slots[0].extension = "retained"
        elif extension == "empty_bank":
            # An empty authored row is invalid for both machines. The new
            # native path must decline rather than manufacturing two slots.
            core._read_cache[(0, 0, 0, 0, 0)] = []
        else:
            original = core._cache_slot
            core._cache_slot = lambda mapping, original=original: original(mapping)
    if extension == "empty_bank":
        for core in pair:
            with pytest.raises(ValueError, match="min.*empty"):
                core.submit(AccessRequest("ordinary", "read", 0, 256 * PAGE))
        return
    submit_pair(pair, AccessRequest("ordinary", "read", 0, 256 * PAGE), native=False)
