"""Boundary histories for NAND physical ordering and compiled page execution."""
import copy

import pytest

from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import (AccessRequest, NandConfig,
                                       NandWearConfig, Operation)
from heterollm_sim.nand_core import NandCore, summarize_nand_observations


PAGE = 1024


def config(**changes):
    values = dict(page_bytes=PAGE, host_granularity_bytes=1,
                  host_bandwidth_gb_s=100., host_read_bandwidth_gb_s=100., host_write_bandwidth_gb_s=100., internal_bandwidth_gb_s=100.,
                  page_read_ns=100., page_program_ns=200., block_erase_ns=2000.)
    values.update(changes)
    return NandConfig(**values)


def req(name, op, address=0, count=PAGE, arrival=0.):
    return AccessRequest(name, op, address, count, arrival)


@pytest.mark.parametrize('details', [False, True])
def test_erase_then_program_waits_for_physical_array_even_with_spare_execution_lane(details):
    core = NandCore(config(),
                    ResourceTimeline(lane_available={'nand:array:0': [0., 0.]}),
                    capture_details=details)
    erased = core.submit(req('erase', Operation.ERASE))
    written = core.submit(req('write', Operation.WRITE))
    assert written.completion_ns >= erased.completion_ns + core.config.page_program_ns
    assert core._array_ready['nand:array:0'] == written.completion_ns
    read = core.submit(req('read', Operation.READ))
    assert read.completion_ns > written.completion_ns
    if details:
        program = next(stage for stage in written.stages if stage.name == 'PAGE_PROGRAM')
        assert program.start_ns >= erased.completion_ns


def test_uninitialized_virtual_array_wire_alias_uses_exact_history_before_acceleration():
    settings = config(channels=2, targets_per_channel=2, planes_per_lun=2,
                      planes_independent=True, parallel_units=1,
                      internal_transfer_bytes=768, host_bandwidth_gb_s=17., host_read_bandwidth_gb_s=17., host_write_bandwidth_gb_s=17.,
                      internal_bandwidth_gb_s=3.1, page_read_ns=91.5,
                      page_program_ns=351., block_erase_ns=901.,
                      metadata={'array_resource_prefix': 'shared',
                                'channel_resource_prefix': 'shared',
                                'host_resource_id': 'shared:0'})
    detailed, summary = NandCore(settings), NandCore(settings, capture_details=False)
    requests = [req('erase', Operation.ERASE),
                req('first-write', Operation.WRITE, count=256 * PAGE),
                req('next-read', Operation.READ, count=256 * PAGE),
                req('next-write', Operation.WRITE, count=256 * PAGE)]
    used_acceleration = False
    for request in requests:
        left, right = detailed.submit(request), summary.submit(request)
        used_acceleration |= right.counters.get('accelerated_pages', 0) > 0
        assert right.completion_ns == pytest.approx(left.completion_ns, rel=1e-11)
        assert right.transfer_bytes == left.transfer_bytes
        assert right.counters['resource_busy_ns'] == pytest.approx(left.counters['resource_busy_ns'])
        assert right.counters['resource_bytes'] == left.counters['resource_bytes']
        assert summary._array_ready == pytest.approx(detailed._array_ready)
        assert summary._buffer_ready == pytest.approx(detailed._buffer_ready)
        for resource, lanes in detailed.timeline.lane_available.items():
            assert summary.timeline.lane_available[resource] == pytest.approx(lanes)
    assert used_acceleration  # Only the first uninitialized alias history falls back.


@pytest.mark.parametrize('buffered', [False, True])
def test_empty_flush_does_not_report_a_time_before_last_observation(buffered):
    core = NandCore(config(write_buffer_bytes=PAGE if buffered else 0))
    first = core.flush_write_buffer(now_ns=1000.)
    earlier = core.flush_write_buffer(now_ns=1.)
    assert first['media_drain_completion_ns'] == 1000.
    assert earlier['media_drain_start_ns'] == earlier['media_drain_completion_ns'] == 1000.
    written = core.submit(req('old-arrival', Operation.WRITE))
    assert all(stage.start_ns >= 1000. for stage in written.stages)


@pytest.mark.parametrize('growth,rated,expected_scale', [
    (0., 3000, 1.), (.5, 10**400, 1.5), (1., 3000, 2.),
])
def test_finite_wear_scaling_does_not_overflow_before_capping_large_integer_counts(growth, rated, expected_scale):
    core = NandCore(config(wear_model=NandWearConfig(initial_pe_cycles=10**400,
                    rated_pe_cycles=rated, program_latency_growth_at_rated_pe=growth,
                    max_program_latency_scale=2., assumption='integer boundary test')))
    written = core.submit(req('write', Operation.WRITE))
    program = next(stage for stage in written.stages if stage.name == 'PAGE_PROGRAM')
    assert program.end_ns - program.start_ns == pytest.approx(core.config.page_program_ns * expected_scale)
    assert written.counters['nand_observations']['events'][0]['pe_cycles_after'] == 10**400
    observed = summarize_nand_observations([written.counters['nand_observations']])
    block = next(iter(observed['wear_blocks'].values()))
    assert block['at_or_above_authored_endurance']
    if rated == 10**400:
        assert block['authored_endurance_fraction_consumed'] == 1.
    else:
        assert block['authored_endurance_fraction_consumed'] is None
        assert block['authored_endurance_fraction_status'] == 'outside_float_range'


def test_finite_wear_ratio_is_not_saturated_by_an_overflowed_intermediate_product():
    core = NandCore(config(wear_model=NandWearConfig(initial_pe_cycles=10**200,
                    rated_pe_cycles=10**300, program_latency_growth_at_rated_pe=1e200,
                    max_program_latency_scale=1e101, assumption='finite-ratio test')))
    written = core.submit(req('write', Operation.WRITE))
    event = written.counters['nand_observations']['events'][0]
    assert event['wear_extra_ns'] == pytest.approx(core.config.page_program_ns * 1e100)


@pytest.mark.parametrize('cover_rest', [False, True])
def test_repeated_staged_payload_does_not_fake_full_page_coverage(cover_rest):
    core = NandCore(config(write_buffer_bytes=2 * PAGE,
                          metadata={}))
    first = core.submit(req('first-half', Operation.WRITE, count=PAGE // 2))
    repeated = core.submit(req('overwrite-first-half', Operation.WRITE, count=PAGE // 2))
    assert first.pages_programmed == repeated.pages_programmed == 0
    if cover_rest:
        core.submit(req('second-half', Operation.WRITE, address=PAGE // 2, count=PAGE // 2))
    assert core.buffered_page_count == 1
    drained = core.flush_write_buffer(now_ns=0.)
    assert drained['physical_programs'] == 1
    assert drained['pages_read'] == int(not cover_rest)
    assert drained['host_transfer_bytes'] == PAGE + (PAGE // 2 if cover_rest else 0)
    assert drained['internal_transfer_bytes'] == PAGE * (1 if cover_rest else 2)


def test_large_multilane_history_falls_back_without_losing_resource_or_byte_state():
    settings = config(channels=2, planes_per_lun=2, planes_independent=True,
                      host_granularity_bytes=1536, internal_transfer_bytes=768)
    lanes = {'nand:array:0': [0., 1000.], 'nand:channel:0': [20., 40.],
             'nand:host': [0., 70.]}
    cores = [NandCore(settings, ResourceTimeline(lane_available={k: list(v) for k, v in lanes.items()}),
                      capture_details=details) for details in (True, False)]
    for request in (req('erase', Operation.ERASE),
                    req('write', Operation.WRITE, 11, 256 * PAGE + 7),
                    req('read', Operation.READ, 17, 256 * PAGE + 23)):
        left, right = [core.submit(request) for core in cores]
        assert right.counters.get('accelerated_pages', 0) == 0
        assert right.completion_ns == left.completion_ns
        assert right.transfer_bytes == left.transfer_bytes
        assert right.counters['resource_bytes'] == left.counters['resource_bytes']
        assert cores[0].timeline.lane_available == cores[1].timeline.lane_available


@pytest.mark.parametrize('entrypoint', ['execute_batch', 'run'])
@pytest.mark.parametrize('invalid', ['partial_rejected', 'buffer_oversize', 'out_of_capacity'])
def test_invalid_batch_is_rejected_before_any_core_or_calendar_mutation(entrypoint, invalid):
    changes = {'partial_page_policy': 'reject'} if invalid == 'partial_rejected' else {}
    if invalid == 'buffer_oversize':
        changes.update(write_buffer_bytes=PAGE, metadata={})
    core = NandCore(config(**changes))
    core.submit(req('history', Operation.READ))
    before = copy.deepcopy(vars(core))
    bad = (req('bad', Operation.WRITE, count=1) if invalid == 'partial_rejected' else
           req('bad', Operation.WRITE, count=2 * PAGE) if invalid == 'buffer_oversize' else
           req('bad', Operation.READ, address=core.config.effective_capacity_bytes))
    with pytest.raises(ValueError):
        getattr(core, entrypoint)(iter([req('valid', Operation.READ), bad]))
    assert vars(core) == before


@pytest.mark.parametrize('shared_array', [False, True])
def test_canonical_aliases_share_existing_wire_reservations_in_detailed_and_accelerated_paths(shared_array):
    aliases = {'nand:host': 'shared-wire', 'nand:channel:0': 'shared-wire',
               'unrelated': 'unrelated-owner'}
    if shared_array:
        aliases['nand:array:0'] = 'shared-wire'
    cores = []
    for details in (True, False):
        timeline = ResourceTimeline(resource_aliases=aliases)
        timeline.reserve('shared-wire', 0., 5000.)
        cores.append(NandCore(config(), timeline, capture_details=details))
    left, right = [core.submit(req('large-write', Operation.WRITE, count=256 * PAGE))
                   for core in cores]
    host = next(stage for stage in left.stages if stage.name == 'HOST_TRANSFER')
    assert host.resource_id == 'shared-wire' and host.start_ns >= 5000.
    assert all(stage.resource_id == 'shared-wire' for stage in left.stages
               if stage.name in ('HOST_TRANSFER', 'INTERNAL_TRANSFER'))
    assert right.counters['accelerated_pages'] > 0
    assert right.completion_ns == pytest.approx(left.completion_ns)
    assert right.counters['resource_bytes'] == left.counters['resource_bytes']
    assert right.counters['resource_busy_ns'] == pytest.approx(left.counters['resource_busy_ns'])
    assert 'nand:host' not in cores[1].timeline.lane_available
    assert cores[1]._array_ready == pytest.approx(cores[0]._array_ready)
