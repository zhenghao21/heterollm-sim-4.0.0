"""The out-of-order generic-NAND controller for submit_many READ batches."""
from dataclasses import replace
from random import Random

import numpy as np
import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_types import AccessRequest
from heterollm_sim._compiled_nand import run_generic_read_batch
from test_generic_nand_resident import logical
from test_nand_generic_buffered_acceleration import PAGE, config
from test_token_owned_history import bits

READ_NS, CHANNEL_NS, HOST_NS = 1000., 40., 1.


def reference(first_pages, counts, arrivals, head, tail, front, geometry, array_ids, channel_ids,
              host_ids, array_ready, buffer_ready, tags, cache_ready, readers, uses, cache_clock,
              starts, lane_counts, lanes, state, moved):
    """Brute-force schedule: repeatedly start the earliest startable job.

    Admission is never credit-limited here. Every server starts a job at
    max(earliest free lane, earliest ready page) and then takes the ready
    page with the smallest (ready time, submission index).
    """
    base_units, planes, independent, per_channel, host_channels = (int(value) for value in geometry)
    pages, bases, admitted = [], [], -np.inf
    for request, (first, count) in enumerate(zip(first_pages, counts)):
        admitted = max(arrivals[request], admitted)
        bases.append(admitted + front)
        for offset in range(count):
            index = first + offset
            base_unit = index % base_units
            bank = base_unit + (index // base_units % planes) * base_units
            host_ns, host_bytes = HOST_NS, 4096
            if offset == 0:
                host_ns, host_bytes = head[request]
            elif offset == count - 1:
                host_ns, host_bytes = tail[request]
            pages.append(dict(request=request, bank=bank, unit=bank if independent else base_unit,
                              channel=base_unit // per_channel, physical=index // (base_units * planes),
                              host_ns=host_ns, host_bytes=host_bytes))
    # Static buffer decisions in submission order, as a sequential controller.
    content = {}
    for number, page in enumerate(pages):
        bank = page["bank"]
        cache_clock += 1
        slot = next((candidate for candidate in range(2) if tags[bank, candidate] == page["physical"]), -1)
        page["hit"] = slot >= 0
        if slot < 0:
            slot = 0
            if ((tags[bank, 1] < 0 and tags[bank, 0] >= 0)
                    or ((tags[bank, 1] < 0) == (tags[bank, 0] < 0) and uses[bank, 1] < uses[bank, 0])):
                slot = 1
        uses[bank, slot] = cache_clock
        key = (bank, slot)
        if key not in content:
            content[key] = dict(fill=None, ready=cache_ready[bank, slot], readers=[], prior=readers[bank, slot])
        if page["hit"]:
            page["source"] = content[key]
            content[key]["readers"].append(number)
        else:
            page["previous"] = content[key]
            content[key] = dict(fill=number, readers=[number], prior=0.)
            page["source"] = content[key]
            tags[bank, slot] = page["physical"]
    queues = {}
    for number, page in enumerate(pages):
        if not page["hit"]:
            queues.setdefault(page["unit"], []).append(number)
    unit_free = {unit: max(array_ready[unit], buffer_ready[unit]) for unit in queues}
    unit_busy = {unit: False for unit in queues}
    times = [dict() for _ in pages]

    def candidates():
        for unit, queue in queues.items():
            if queue and not unit_busy[unit]:
                number = queue[0]
                yield array_ids[unit], number, max(bases[pages[number]["request"]], unit_free[unit]), 0
        for number, page in enumerate(pages):
            row = times[number]
            if not page["hit"] and "read_end" in row and "channel" not in row:
                previous = page["previous"]
                done = [times[reader].get("host_end") for reader in previous["readers"]]
                if None not in done:
                    ready = max([row["read_end"], previous["prior"], *done])
                    yield channel_ids[page["channel"]], number, ready, 1
            if "host" not in row:
                source = page["source"]
                host = host_ids[page["channel"] % host_channels]
                if page["hit"]:
                    fill = source["fill"]
                    if fill is None:
                        yield host, number, max(bases[page["request"]], source["ready"]), 2
                    elif "channel_end" in times[fill]:
                        yield host, number, max(bases[page["request"]], times[fill]["channel_end"]), 2
                elif "channel_end" in row:
                    yield host, number, row["channel_end"], 2

    remaining = 3 * sum(not page["hit"] for page in pages) + sum(page["hit"] for page in pages)
    while remaining:
        by_server = {}
        for rid, number, ready, stage in candidates():
            by_server.setdefault(rid, []).append((ready, number, stage))
        best = None
        for rid, jobs in by_server.items():
            first = starts[rid]
            lane = min(range(first, first + lane_counts[rid]), key=lambda at: (lanes[at], at))
            start = max(lanes[lane], min(ready for ready, _, _ in jobs))
            ready, number, stage = min(job for job in jobs if job[0] <= start)
            if best is None or start < best[0]:
                best = (start, rid, lane, number, stage)
        start, rid, lane, number, stage = best
        page, row = pages[number], times[number]
        duration, payload, direction = ((READ_NS, -1, 0), (CHANNEL_NS, 4096, 1),
                                        (page["host_ns"], page["host_bytes"], 1))[stage]
        end = start + duration
        lanes[lane] = end
        state[rid, 0] = min(lanes[starts[rid]:starts[rid] + lane_counts[rid]])
        state[rid, 1] += duration
        state[rid, 2], state[rid, 3] = start, end
        if payload >= 0:
            moved[rid] += payload
        if stage == 0:
            unit = page["unit"]
            queues[unit].pop(0)
            unit_busy[unit] = True
            row["read_end"] = end
        elif stage == 1:
            row["channel"], row["channel_end"] = start, end
            unit = page["unit"]
            unit_busy[unit], unit_free[unit] = False, end
            array_ready[unit], buffer_ready[unit] = row["read_end"], end
        else:
            row["host"], row["host_end"] = start, end
        remaining -= 1
    for (bank, slot), generation in content.items():
        ends = [times[reader]["host_end"] for reader in generation["readers"]]
        readers[bank, slot] = max([generation["prior"], *ends])
        if generation["fill"] is not None:
            cache_ready[bank, slot] = times[generation["fill"]]["channel_end"]
    completions = []
    for request in range(len(first_pages)):
        ends = [times[number]["host_end"] for number, page in enumerate(pages) if page["request"] == request]
        completions.append(max(bases[request], *ends))
    return completions


def random_case(seed):
    rng = Random(seed)
    base_units, planes, per_channel = 4, rng.choice((1, 2, 3)), 2
    independent = rng.random() < .7
    host_channels = rng.choice((1, 2))
    banks = base_units * planes
    units = banks if independent else base_units
    channels = base_units // per_channel
    array_ids = np.arange(units, dtype=np.int64)
    channel_ids = units + np.arange(channels, dtype=np.int64)
    host_ids = units + channels + np.arange(host_channels, dtype=np.int64)
    resources = units + channels + host_channels
    lane_counts = np.array([rng.choice((1, 1, 2)) for _ in range(resources)], dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(lane_counts)[:-1])).astype(np.int64)
    lanes = np.array([rng.choice((0., 0., 250., 1500.)) for _ in range(int(lane_counts.sum()))])
    state = np.zeros((resources, 4))
    for rid in range(resources):
        state[rid, 0] = lanes[starts[rid]:starts[rid] + lane_counts[rid]].min()
    tags = np.full((banks, 2), -1, dtype=np.int64)
    uses = np.zeros((banks, 2), dtype=np.int64)
    ready = np.zeros((banks, 2))
    readers = np.zeros((banks, 2))
    for bank in range(banks):
        for slot in range(2):
            if rng.random() < .4:
                tags[bank, slot] = rng.randrange(3)
                uses[bank, slot] = rng.randrange(5)
                ready[bank, slot] = rng.choice((0., 300.))
                readers[bank, slot] = rng.choice((0., 600., 2500.))
    requests = []
    for _ in range(rng.randrange(1, 6)):
        count = rng.randrange(1, 7)
        head_bytes, tail_bytes = rng.choice((256, 4096)), rng.choice((512, 4096))
        requests.append((rng.randrange(3 * banks), count, rng.choice((0., 0., 100., 2000.)),
                         (head_bytes / 4096, head_bytes), (tail_bytes / 4096, tail_bytes)))
    return dict(
        requests=requests, front=rng.choice((0., 5.)),
        geometry=np.array((base_units, planes, independent, per_channel, host_channels), dtype=np.int64),
        ids=(array_ids, channel_ids, host_ids),
        controller=(np.array([rng.choice((0., 900.)) for _ in range(units)]),
                    np.array([rng.choice((0., 1100.)) for _ in range(units)]), tags, ready, readers, uses),
        calendar=(starts, lane_counts, lanes, state, np.zeros(resources, dtype=np.int64)))


@pytest.mark.parametrize("seed", range(60))
def test_kernel_matches_brute_force_schedule(seed):
    case = random_case(seed)
    requests, (array_ids, channel_ids, host_ids) = case["requests"], case["ids"]
    first_pages = np.array([row[0] for row in requests], dtype=np.int64)
    counts = np.array([row[1] for row in requests], dtype=np.int64)
    arrivals = np.array([row[2] for row in requests])
    head, tail = [row[3] for row in requests], [row[4] for row in requests]
    expected_controller = tuple(value.copy() for value in case["controller"])
    expected_calendar = tuple(value.copy() for value in case["calendar"])
    expected = reference(first_pages.tolist(), counts.tolist(), arrivals.tolist(), head, tail, case["front"],
                         case["geometry"], array_ids, channel_ids, host_ids, *expected_controller, 0,
                         *expected_calendar)
    arrays, buffers, tags, ready, readers, uses = case["controller"]
    starts, lane_counts, lanes, state, moved = case["calendar"]
    banks = len(tags)
    page_indices = np.concatenate([np.arange(first, first + count, dtype=np.int64)
                                   for first, count in zip(first_pages.tolist(), counts.tolist())])
    output = run_generic_read_batch(
        page_indices, counts, arrivals,
        np.array([row[1] for row in head], dtype=np.int64), np.array([row[0] for row in head]),
        np.array([row[1] for row in tail], dtype=np.int64), np.array([row[0] for row in tail]),
        np.full(int(counts.sum()), -1.), 0., False,
        np.zeros(len(counts), dtype=np.int64), np.zeros(1), np.empty(0), np.zeros(2, dtype=np.int64),
        1 << 30, case["front"], case["geometry"],
        np.array((READ_NS, CHANNEL_NS, HOST_NS, 4096.)), array_ids, channel_ids, host_ids, host_ids,
        arrays, buffers, tags, ready, readers, uses, np.zeros(banks, dtype=np.bool_), 0,
        starts, lane_counts, lanes, np.zeros(len(lanes), dtype=np.int64), state, moved,
        np.zeros(len(starts), dtype=np.int64))
    assert bits(output[1].tolist()) == bits([float(value) for value in expected])
    for actual, wanted in zip((arrays, buffers, tags, ready, readers, uses, lanes, state, moved),
                              (*expected_controller, expected_calendar[2], expected_calendar[3],
                               expected_calendar[4])):
        assert bits(actual.tolist()) == bits(wanted.tolist())


def runtime(**changes):
    context = PhysicalRuntimeContext(config(**changes), "flash", capture_details=False)
    core = context.runtimes["flash"].core
    core.submit(AccessRequest("seed", "read", 0, 128 * PAGE))
    assert core._nand_numeric_state is not None
    return context, core


def test_busy_plane_no_longer_holds_back_other_requests_on_its_channel():
    # Read-bound geometry: 5 us array reads, ~10 ns channel pages. Banks 0
    # and 1 share channel 0; bank 0 is busy with an earlier request.
    settings = dict(internal_bandwidth_gb_s=400., page_read_ns=5000.)
    (_, batch_core), (_, ordered_core) = runtime(**settings), runtime(**settings)
    banks = batch_core.config.channels * batch_core.config.targets_per_channel * 2 * 3
    start = 1e6
    busy = AccessRequest("busy", "read", 3 * banks * PAGE, PAGE, start)
    batch = [AccessRequest("late", "read", 4 * banks * PAGE, PAGE, start),
             AccessRequest("free", "read", 4 * banks * PAGE + PAGE, PAGE, start)]
    for core in (batch_core, ordered_core):
        core.submit(busy)
    late, free = batch_core.submit_many(batch)
    ordered_late, ordered_free = (ordered_core.submit(request) for request in batch)
    # The free plane's page leaves as soon as its read ends; one request at
    # a time it queued behind the busy plane's page on their shared channel.
    assert free.completion_ns < start + 6000 < 2 * 5000 + start < late.completion_ns
    assert ordered_free.completion_ns > late.completion_ns
    assert late.completion_ns == ordered_late.completion_ns


@pytest.mark.parametrize("depth", [1, 3])
def test_credits_are_taken_in_order_and_match_the_host_queue(depth):
    context, core = runtime(max_outstanding_requests=depth)
    prior = [end for end, _ in core.__dict__["_inflight_operations"]]
    acceptance = core._acceptance_ns
    requests = [AccessRequest(str(index), "read", (index % 5) * 37 * PAGE + 17, 33 * PAGE - 23, .125 * index)
                for index in range(12)]
    results = core.submit_many(requests)
    waits = [row.counters["queue_wait_ns"] for row in results]
    assert any(waits)
    # Replay the host queue from the scheduled completions: each request
    # takes the first credit free at or after its arrival, in order.
    held = list(prior)
    for row, wait in zip(results, waits):
        effective = max(row.arrival_ns, acceptance)
        while sum(end > effective for end in held) >= depth:
            effective = min(end for end in held if end > effective)
        assert row.arrival_ns + wait == effective
        assert row.counters["host_outstanding_at_admission_peak"] == sum(end > effective for end in held) + 1
        held.append(row.completion_ns)
        acceptance = effective


def test_hits_inside_one_batch_wait_for_their_fill_and_bill_no_media():
    context, core = runtime()
    address = 9 * PAGE * core.config.channels * core.config.planes_per_lun
    first, again = core.submit_many([AccessRequest("fill", "read", address, 4 * PAGE),
                                     AccessRequest("hit", "read", address, 4 * PAGE)])
    assert first.counters["read_cache_misses"] == 4 and again.counters["read_cache_hits"] == 4
    assert again.counters["pages_read"] == 0 and "resource_busy_ns" in again.counters
    assert all("array" not in rid for rid in again.counters["resource_busy_ns"])
    assert again.completion_ns > first.completion_ns - 4 * READ_NS


def test_sink_and_request_maps_bill_the_same_bits():
    pair = [runtime(host_granularity_bytes=256)[1] for _ in range(2)]
    requests = [AccessRequest(str(index), "read", (index * 41 + 3) * PAGE + 17, (index % 4 + 2) * PAGE - 33)
                for index in range(20)]
    plain = pair[0].submit_many(requests)
    sink = tuple({} for _ in range(6))
    with_sink = pair[1].submit_many(requests, _resource_sink=sink)
    expected = ({}, {}, {})
    for row in plain:
        for rid, value in row.counters["resource_busy_ns"].items():
            expected[0][rid] = expected[0].get(rid, 0.0) + value
        for rid, value in row.counters["resource_bytes"].items():
            expected[1][rid] = expected[1].get(rid, 0) + value
        for rid, value in row.counters["resource_last_intervals"].items():
            previous = expected[2].get(rid)
            if previous is None or (value[1], value[0]) > (previous[1], previous[0]):
                expected[2][rid] = value
    assert bits(sink[:3]) == bits(expected) == bits(sink[3:])
    stripped = [replace(row, counters={key: value for key, value in row.counters.items()
                                       if not key.startswith("resource_")}) for row in plain]
    assert bits([replace(row, counters={key: value for key, value in row.counters.items()
                                        if not key.startswith("resource_")}) for row in with_sink]) == bits(stripped)
    assert bits(logical(pair[0])) == bits(logical(pair[1]))


@pytest.mark.parametrize("finish", ["commit_transaction", "restore"])
def test_rollback_restores_the_state_before_the_batch(finish):
    context, core = runtime(max_outstanding_requests=3)
    before = bits(logical(core))
    outer = context.begin_transaction(owners=("flash",), _owned_timeline=True)
    core.submit(AccessRequest("prefix", "write", 33, 128 * PAGE + 7))
    inner = context.begin_transaction(owners=("flash",), _owned_timeline=True)
    core.submit_many([AccessRequest(str(index), "read", index * PAGE + 17, 40 * PAGE) for index in range(9)])
    getattr(context, finish)(inner)
    context.restore(outer)
    assert bits(logical(core)) == before


def test_invalid_or_overflowing_batch_changes_nothing():
    context, core = runtime()
    before = bits(logical(core))
    with pytest.raises(ValueError):
        core.submit_many([AccessRequest("valid", "read", 17, 128 * PAGE),
                          AccessRequest("invalid", "read", core.config.effective_capacity_bytes, PAGE)])
    assert bits(logical(core)) == before
    core.config = replace(core.config, page_read_ns=1e308)
    banks = core.config.channels * core.config.targets_per_channel * 2 * 3
    with pytest.raises(ValueError, match="finite timing range"):
        core.submit_many([AccessRequest(str(index), "read", (index + 9) * banks * PAGE, PAGE)
                          for index in range(3)])
    assert bits(logical(core)) == before
