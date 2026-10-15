"""Cross-request staging readers and media completion semantics."""
import copy

import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext, _PhysicalRuntime
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest, NandConfig, Operation
from heterollm_sim.nand_core import NandCore


PAGE = 1024


def request(name, op, address=0, amount=PAGE, arrival=0.):
    return AccessRequest(name, op, address, amount, arrival)


def config(**updates):
    values = dict(page_bytes=PAGE, host_granularity_bytes=1,
                  host_bandwidth_gb_s=100., host_read_bandwidth_gb_s=100., host_write_bandwidth_gb_s=100., internal_bandwidth_gb_s=100.,
                  page_read_ns=10., page_program_ns=20., block_erase_ns=100.)
    values.update(updates)
    return NandConfig(**values)


@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('details', [False, True])
def test_flush_keeps_staging_reader_source_alive_without_inflating_media_completion(partial, details):
    core = NandCore(config(host_read_bandwidth_gb_s=1., write_buffer_bytes=PAGE,
                          metadata={}),
                    ResourceTimeline(lane_available={'nand:host': [0., 0.]}),
                    capture_details=details)
    amount = PAGE // 2 if partial else PAGE
    core.submit(request('stage', Operation.WRITE, amount=amount))
    reader = core.submit(request('reader', Operation.READ))
    assert reader.counters['staging_read_merges'] == int(partial)
    assert reader.staging_read_hits == int(not partial)
    drain = core.flush_write_buffer(now_ns=0.)
    assert drain['media_drain_completion_ns'] < reader.completion_ns
    assert drain['write_buffer_release_completion_ns'] == reader.completion_ns
    assert core.inflight_write_buffer_bytes == amount
    repeat = core.flush_write_buffer(now_ns=drain['media_drain_completion_ns'])
    assert repeat['physical_programs'] == 0
    assert repeat['media_drain_completion_ns'] == drain['media_drain_completion_ns']
    assert repeat['write_buffer_release_completion_ns'] == reader.completion_ns
    reused = core.submit(request('reuse', Operation.WRITE, address=PAGE))
    assert reused.completion_ns > reader.completion_ns
    assert reused.physical_write_bytes == reused.physical_read_bytes == 0
    assert reused.write_buffer_occupancy_bytes == PAGE


def test_reader_retention_survives_runtime_rollback_and_auto_drain_charges_media_once():
    runtime = PhysicalRuntimeContext()
    runtime.timeline.lane_available['nand:host'] = [0., 0.]
    core = NandCore(config(host_read_bandwidth_gb_s=1., write_buffer_bytes=PAGE,
                          metadata={}), runtime.timeline)
    runtime.runtimes['nand'] = _PhysicalRuntime(core)
    core.submit(request('stage', Operation.WRITE))
    reader = core.submit(request('reader', Operation.READ))
    saved = runtime.snapshot()
    first = core.submit(request('auto-drain', Operation.WRITE, address=PAGE))
    runtime.restore(saved)
    retry = core.submit(request('auto-drain', Operation.WRITE, address=PAGE))
    assert first == retry
    assert retry.write_buffer_wait_ns == reader.completion_ns
    assert retry.physical_write_bytes == PAGE
    assert retry.pages_programmed == 1 and retry.pages_read == 0
    assert retry.counters['auto_drain_physical_programs'] == 1


def test_staging_reader_retention_does_not_prevent_other_pages_with_spare_capacity():
    core = NandCore(config(host_read_bandwidth_gb_s=1., write_buffer_bytes=2 * PAGE,
                          metadata={}),
                    ResourceTimeline(lane_available={'nand:host': [0., 0.]}))
    core.submit(request('stage', Operation.WRITE))
    reader = core.submit(request('reader', Operation.READ))
    core.flush_write_buffer(now_ns=0.)
    unrelated = core.submit(request('spare-buffer', Operation.WRITE, address=PAGE))
    assert unrelated.completion_ns < reader.completion_ns
    assert unrelated.write_buffer_wait_ns == 0.
    assert unrelated.write_buffer_occupancy_bytes == 2 * PAGE


def test_discard_only_removes_selected_dirty_pages_and_retains_prior_transfer_costs():
    core = NandCore(config(write_buffer_bytes=4 * PAGE, metadata={}))
    core.submit(request('page0', Operation.WRITE, amount=PAGE // 2))
    core.submit(request('page0-again', Operation.WRITE, amount=PAGE // 4))
    core.submit(request('page1', Operation.WRITE, address=PAGE))
    core.submit(request('page2', Operation.WRITE, address=2 * PAGE))
    before_bytes, before_busy = dict(core.timeline.bytes_moved), dict(core.timeline.busy_ns)
    discarded = core.discard_staged_pages(0, 2 * PAGE, now_ns=0.)
    assert discarded['discarded_pages'] == 2
    assert discarded['discarded_host_bytes'] == discarded['discarded_logical_bytes'] == 7 * PAGE // 4
    assert discarded['discarded_merge_bytes'] == PAGE // 4
    assert discarded['physical_programs'] == discarded['host_transfer_bytes'] == 0
    assert core.timeline.bytes_moved == before_bytes and core.timeline.busy_ns == before_busy
    assert core.buffered_page_count == 1 and core.pending_write_buffer_bytes == PAGE
    assert core.inflight_write_buffer_bytes == 7 * PAGE // 4
    drained = core.flush_write_buffer(now_ns=0.)
    assert drained['flushed_pages'] == drained['physical_programs'] == 1
    assert drained['physical_write_bytes'] == PAGE and drained['merge_bytes'] == 0


@pytest.mark.parametrize('reader', [False, True])
def test_discard_preserves_inflight_host_source_until_payload_and_readers_finish(reader):
    core = NandCore(config(host_read_bandwidth_gb_s=1., write_buffer_bytes=PAGE,
                          metadata={}),
                    ResourceTimeline(lane_available={'nand:host': [0., 0.]}))
    write = core.submit(request('stage', Operation.WRITE))
    finished = core.submit(request('read', Operation.READ)).completion_ns if reader else write.completion_ns
    discarded = core.discard_staged_pages(0, PAGE, now_ns=0.)
    assert discarded['write_buffer_release_completion_ns'] == finished
    assert core.buffered_page_count == core.pending_write_buffer_bytes == 0
    assert core.inflight_write_buffer_bytes == PAGE
    empty = core.flush_write_buffer(now_ns=0.)
    assert empty['physical_programs'] == 0 and empty['media_drain_completion_ns'] == 0.
    assert empty['write_buffer_release_completion_ns'] == finished
    replacement = core.submit(request('replacement', Operation.WRITE, address=PAGE))
    assert replacement.write_buffer_wait_ns == finished
    assert replacement.pages_programmed == replacement.pages_read == 0
    assert core.flush_write_buffer()['physical_programs'] == 1


def test_discard_does_not_revoke_already_scheduled_media_work():
    core = NandCore(config(write_buffer_bytes=PAGE, metadata={}))
    core.submit(request('stage', Operation.WRITE))
    drained = core.flush_write_buffer(now_ns=0.)
    before = copy.deepcopy(vars(core))
    assert core.discard_staged_pages(0, PAGE)['discarded_pages'] == 0
    assert vars(core) == before
    replacement = core.submit(request('replacement', Operation.WRITE, address=PAGE))
    assert replacement.write_buffer_wait_ns == drained['media_drain_completion_ns']


@pytest.mark.parametrize('address,amount', [(1, PAGE), (0, PAGE - 1), (-PAGE, PAGE), (0, 0), (True, PAGE)])
def test_discard_invalid_range_rejects_without_mutating_dirty_or_calendar_state(address, amount):
    core = NandCore(config(write_buffer_bytes=PAGE, metadata={}))
    core.submit(request('stage', Operation.WRITE))
    before = copy.deepcopy(vars(core))
    with pytest.raises(ValueError, match='whole aligned NAND pages'):
        core.discard_staged_pages(address, amount, now_ns=1000.)
    assert vars(core) == before


def test_discard_on_write_through_core_is_harmless():
    core = NandCore(config())
    core.submit(request('write-through', Operation.WRITE))
    before = copy.deepcopy(vars(core))
    result = core.discard_staged_pages(0, PAGE)
    assert result['discarded_pages'] == result['physical_programs'] == 0
    assert vars(core) == before
