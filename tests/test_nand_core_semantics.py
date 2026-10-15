"""NAND operations, resource dependencies and accounting use actual Core work."""
import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext, _PhysicalRuntime
from heterollm_sim.memory_types import (AccessRequest, NandConfig,
                                       NandWearConfig, Operation)
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.nand_core import NandCore


PAGE = 4096


def config(**changes):
    values = dict(channels=1, planes_per_lun=1, blocks_per_plane=4,
                  pages_per_block=512, page_bytes=PAGE, host_granularity_bytes=1,
                  host_bandwidth_gb_s=100., host_read_bandwidth_gb_s=100., host_write_bandwidth_gb_s=100., internal_bandwidth_gb_s=10.,
                  page_read_ns=100., page_program_ns=1000., block_erase_ns=2000.)
    values.update(changes)
    return NandConfig(**values)


def access(name, operation, address=0, count=PAGE, arrival=0.):
    return AccessRequest(name, operation, address, count, arrival)


def test_parallel_channels_are_included_in_device_bandwidth_ceiling():
    core = NandCore(config(channels=4, internal_bandwidth_gb_s=1., page_read_ns=0.,
                          host_read_bandwidth_gb_s=50., host_write_bandwidth_gb_s=2.))
    read = core.submit(access('read', Operation.READ, count=128 * PAGE))
    assert 3.9 < read.logical_bytes / read.completion_ns <= read.bandwidth_ceiling_gb_s == 4.
    assert {stage.resource_id for stage in read.stages if stage.name == 'INTERNAL_TRANSFER'} == {
        'nand:channel:0', 'nand:channel:1', 'nand:channel:2', 'nand:channel:3'}
    write = core.submit(access('write', Operation.WRITE, arrival=read.completion_ns))
    assert write.bandwidth_ceiling_gb_s == 2.


def test_partial_write_across_page_boundary_charges_two_reads_and_programs():
    core = NandCore(config(host_granularity_bytes=64))
    result = core.submit(access('boundary', Operation.WRITE, PAGE - 1, 2))
    assert result.logical_bytes == 2
    assert result.pages_read == result.pages_programmed == 2
    assert result.physical_read_bytes == result.physical_write_bytes == 2 * PAGE
    assert result.internal_transfer_bytes == result.transfer_bytes == 4 * PAGE
    assert result.host_transfer_bytes == 128
    assert result.counters['rmw_read_operations'] == 2
    reads = [stage for stage in result.stages if stage.name == 'RMW_PAGE_READ']
    programs = [stage for stage in result.stages if stage.name == 'PAGE_PROGRAM']
    assert all(program.start_ns >= read.end_ns for read, program in zip(reads, programs))


def test_staging_ack_and_read_report_host_path_ceiling_and_batch_uses_it():
    core = NandCore(config(internal_bandwidth_gb_s=1., write_buffer_bytes=PAGE,
                          host_write_bandwidth_gb_s=50., host_read_bandwidth_gb_s=25.,
                          metadata={}))
    batch = core.run([access('stage', Operation.WRITE), access('read-stage', Operation.READ)])
    write, read = batch.requests
    assert write.internal_transfer_bytes == read.internal_transfer_bytes == 0
    assert write.logical_bytes / write.completion_ns == write.bandwidth_ceiling_gb_s == 50.
    assert read.bandwidth_ceiling_gb_s == 25.
    assert read.logical_bytes / (read.completion_ns - write.completion_ns) == pytest.approx(25.)
    assert 'host acknowledgement is not media drain' in write.counters['bandwidth_ceiling_scope']
    assert batch.bandwidth_ceiling_gb_s == batch.write_bandwidth_ceiling_gb_s == 50.
    assert batch.read_bandwidth_ceiling_gb_s == 25.
    core.flush_write_buffer()
    media_read = core.submit(access('read-media', Operation.READ))
    assert media_read.internal_transfer_bytes == PAGE
    assert media_read.bandwidth_ceiling_gb_s == 1.


def test_striped_erase_deduplicates_physical_blocks_and_shared_planes_serialize():
    core = NandCore(config(channels=2, planes_per_lun=2, planes_independent=False,
                          pages_per_block=2))
    result = core.submit(access('erase', Operation.ERASE, count=6 * PAGE))
    assert result.erase_operations == 4
    assert len({(m.channel, m.plane, m.block) for m in result.mapping}) == 4
    assert result.logical_bytes == result.transfer_bytes == 0
    assert result.completion_ns == 2 * core.config.block_erase_ns
    for resource in ('nand:array:0', 'nand:array:1'):
        stages = [stage for stage in result.stages if stage.resource_id == resource]
        assert len(stages) == 2 and stages[1].start_ns == stages[0].end_ns


@pytest.mark.parametrize('changes', [
    dict(channels=4, planes_per_lun=2, planes_independent=True),
    dict(channels=2, planes_per_lun=3, planes_independent=False, parallel_units=1,
         host_granularity_bytes=1000),
    dict(channels=2, targets_per_channel=2, planes_per_lun=2, planes_independent=True,
         parallel_units=3, host_granularity_bytes=8192),
])
def test_accelerated_mixed_stream_preserves_detailed_timing_bytes_and_future_state(changes):
    detailed = NandCore(config(**changes))
    accelerated = NandCore(detailed.config, capture_details=False)
    requests = [access('large-read', Operation.READ, 11, 2048 * PAGE + 93),
                access('large-write', Operation.WRITE, 33, 2048 * PAGE + 57),
                access('later-partial', Operation.WRITE, 7, 13),
                access('later-read', Operation.READ, PAGE, PAGE)]
    saw_acceleration = False
    for request in requests:
        left, right = detailed.submit(request), accelerated.submit(request)
        saw_acceleration |= right.counters.get('accelerated_pages', 0) > 0
        assert right.completion_ns == pytest.approx(left.completion_ns, rel=1e-11)
        for field in ('logical_bytes', 'transfer_bytes', 'pages_read', 'pages_programmed',
                      'host_transfer_bytes', 'internal_transfer_bytes',
                      'physical_read_bytes', 'physical_write_bytes'):
            assert getattr(left, field) == getattr(right, field)
        assert right.counters['resource_busy_ns'] == pytest.approx(left.counters['resource_busy_ns'])
        assert right.counters['resource_bytes'] == left.counters['resource_bytes']
    assert saw_acceleration
    assert accelerated._array_ready == pytest.approx(detailed._array_ready)
    assert accelerated._buffer_ready == pytest.approx(detailed._buffer_ready)
    assert accelerated.timeline.ready_ns == pytest.approx(detailed.timeline.ready_ns)


def test_explicit_flush_retains_source_buffer_until_media_completion():
    core = NandCore(config(internal_bandwidth_gb_s=1., write_buffer_bytes=PAGE,
                          metadata={}))
    first = core.submit(access('first', Operation.WRITE))
    drain = core.flush_write_buffer(now_ns=0.)
    assert drain['media_drain_start_ns'] >= first.completion_ns
    assert core.pending_write_buffer_bytes == core.buffered_page_count == 0
    assert core.inflight_write_buffer_bytes == PAGE
    repeated = core.flush_write_buffer(now_ns=0.)
    assert repeated['physical_programs'] == 0
    assert repeated['media_drain_completion_ns'] == drain['media_drain_completion_ns']
    second = core.submit(access('second', Operation.WRITE, PAGE))
    host = next(stage for stage in second.stages if stage.name == 'HOST_TRANSFER')
    assert host.start_ns >= drain['media_drain_completion_ns']
    assert second.write_buffer_wait_ns == drain['media_drain_completion_ns']
    assert second.write_buffer_occupancy_bytes == PAGE
    assert core.inflight_write_buffer_bytes == 0
    # The prior explicit drain owns its physical cost; admission must not bill it again.
    assert second.physical_read_bytes == second.physical_write_bytes == 0


def test_late_empty_flush_releases_completed_drains_and_preserves_time_order():
    core = NandCore(config(write_buffer_bytes=PAGE, metadata={}))
    core.submit(access('first', Operation.WRITE))
    drain = core.flush_write_buffer(now_ns=0.)
    observed_now = drain['media_drain_completion_ns'] + 100.
    empty = core.flush_write_buffer(now_ns=observed_now)
    assert empty['physical_programs'] == 0
    assert empty['media_drain_completion_ns'] == observed_now
    assert core.inflight_write_buffer_bytes == 0
    second = core.submit(access('second', Operation.WRITE, PAGE))
    assert min(stage.start_ns for stage in second.stages) >= observed_now


def test_erase_triggered_drain_retains_rmw_counter_without_double_charging():
    core = NandCore(config(write_buffer_bytes=PAGE, metadata={}))
    staged = core.submit(access('partial', Operation.WRITE, count=1024))
    erased = core.submit(access('erase', Operation.ERASE))
    assert staged.pages_programmed == staged.pages_read == 0
    assert erased.pages_read == erased.counters['rmw_read_operations'] == 1
    assert erased.pages_programmed == erased.erase_operations == 1
    assert erased.internal_transfer_bytes == 2 * PAGE
    assert core.flush_write_buffer()['physical_programs'] == 0
    assert core.inflight_write_buffer_bytes == core.pending_write_buffer_bytes == 0


@pytest.mark.parametrize('overlap', [True, False])
def test_partial_staging_read_waits_for_dirty_overlay_only_when_ranges_overlap(overlap):
    timeline = ResourceTimeline(lane_available={'nand:host': [0., 0.]})
    core = NandCore(config(host_write_bandwidth_gb_s=1., internal_bandwidth_gb_s=100.,
                          write_buffer_bytes=PAGE, metadata={}), timeline)
    write = core.submit(access('slow-dirty-payload', Operation.WRITE, count=1024))
    read = core.submit(access('read', Operation.READ, 0 if overlap else 2048,
                             PAGE if overlap else 1024))
    assert read.pages_read == 1 and read.staging_read_hits == 0
    assert read.counters['staging_read_merges'] == int(overlap)
    host = next(stage for stage in read.stages if stage.name == 'HOST_TRANSFER')
    if overlap:
        assert host.start_ns >= write.completion_ns
    else:
        assert read.completion_ns < write.completion_ns


def test_snapshot_replay_restores_staged_drains_feedback_and_resource_costs():
    runtime = PhysicalRuntimeContext()
    core = NandCore(config(write_buffer_bytes=PAGE, metadata={},
                          wear_model=NandWearConfig(assumption='synthetic test')), runtime.timeline)
    runtime.runtimes['nand'] = _PhysicalRuntime(core)
    core.submit(access('first', Operation.WRITE, count=1024))
    drain = core.flush_write_buffer(now_ns=0.)
    snapshot = runtime.snapshot()

    def continuation():
        write = core.submit(access('second', Operation.WRITE, PAGE, 1024))
        erase = core.submit(access('erase', Operation.ERASE, arrival=write.completion_ns))
        return write, erase, dict(core.timeline.busy_ns), dict(core.timeline.bytes_moved)

    original = continuation()
    runtime.restore(snapshot)
    assert core.inflight_write_buffer_bytes == 1024
    assert core.flush_write_buffer(now_ns=0.)['media_drain_completion_ns'] == drain['media_drain_completion_ns']
    replay = continuation()
    assert original == replay
