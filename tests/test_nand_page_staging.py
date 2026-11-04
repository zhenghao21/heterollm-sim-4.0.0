"""Partial-page staging: KV tails wait in the intermediate store until full."""
from dataclasses import replace
import random

import pytest

from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest, NandConfig, Operation, SramConfig
from heterollm_sim.nand_core import NandCore
from heterollm_sim.sram_core import SramCore

PAGE = 4096


def config(mode="logic_die", capacity=1 << 20, **changes):
    base = NandConfig(kind="HBF", channels=8, planes_per_lun=2, planes_independent=True, page_bytes=PAGE,
        host_granularity_bytes=PAGE, host_bandwidth_gb_s=100., host_read_bandwidth_gb_s=100.,
        host_write_bandwidth_gb_s=100., internal_bandwidth_gb_s=10., page_read_ns=1000.,
        page_program_ns=100000., blocks_per_plane=64, pages_per_block=64, read_cache_pages_per_bank=2,
        host_channels=4, host_full_duplex=True, max_outstanding_requests=64,
        capacity_bytes=8 * 2 * 64 * 64 * PAGE, page_staging=mode,
        page_staging_bytes_per_host_channel=capacity, page_staging_latency_ns=1.)
    return replace(base, **changes)


def core(mode="logic_die", numeric=True, **changes):
    nand = NandCore(config(mode, **changes), capture_details=not numeric)
    nand._nand_numeric_enabled = numeric
    return nand


def write(nand, name, address, size, at):
    return nand.submit(AccessRequest(name, Operation.WRITE, address, size, at))


def read(nand, name, address, size, at):
    return nand.submit(AccessRequest(name, Operation.READ, address, size, at))


@pytest.mark.parametrize("numeric", [False, True])
def test_rows_are_acknowledged_from_the_store_and_a_full_page_programs_without_rmw(numeric):
    nand = core(numeric=numeric)
    rows = [write(nand, f"row{index}", 10 * PAGE + index * 1024, 1024, index * 1000.) for index in range(4)]
    # 1 KiB over one 25 GB/s stack channel, then the 1 ns store latency.
    assert [row.completion_ns - row.arrival_ns for row in rows] == [pytest.approx(41.96)] * 4
    assert [row.pages_programmed for row in rows] == [0, 0, 0, 1]
    assert all(row.physical_read_bytes == 0 for row in rows)
    page_channel = 10 % 8 % 4
    state = nand.page_staging_state()
    assert state["valid_bytes"][page_channel] == PAGE  # held until the program ends
    program_end = max(nand.timeline.ready_ns[name] for name in nand.timeline.ready_ns if ":array:" in name
                      or name.startswith("nand:array"))
    assert program_end > rows[-1].completion_ns + 100000.
    # Any later staging call at or after the program end releases the page.
    write(nand, "next", 11 * PAGE, 1024, program_end)
    assert nand.page_staging_state()["valid_bytes"][page_channel] == 0
    assert nand.page_staging_state()["peak_bytes"][page_channel] == PAGE


@pytest.mark.parametrize("numeric", [False, True])
def test_reads_take_staged_pages_from_the_store_until_their_program_completes(numeric):
    nand = core(numeric=numeric)
    for index in range(6):
        write(nand, f"row{index}", 10 * PAGE + index * 1024, 1024, index * 1000.)
    early = read(nand, "early", 10 * PAGE, 6 * 1024, 6000.)
    assert (early.staging_read_hits, early.pages_read) == (2, 0)
    # A full page and a 2 KiB tail, both from the store over their channels.
    assert early.completion_ns - early.arrival_ns == pytest.approx(1. + PAGE / 25.)
    late = read(nand, "late", 10 * PAGE, 6 * 1024, 300000.)
    assert (late.staging_read_hits, late.pages_read) == (1, 1)


def test_scalar_and_native_controllers_agree_on_a_random_kv_stream():
    rng = random.Random(7)
    streams = [(base, 0) for base in (3 * PAGE, 40 * PAGE, 77 * PAGE)]
    events, now = [], 0.
    for step in range(40):
        now += rng.choice((50., 500., 5000., 150000.))
        index = rng.randrange(len(streams))
        base, tokens = streams[index]
        events.append(("w", base + tokens * 512, 512, now))
        streams[index] = (base, tokens + 1)
        if step % 3 == 2:
            base, tokens = streams[rng.randrange(len(streams))]
            if tokens:
                events.append(("r", base, tokens * 512, now + 1.))
    results = []
    for numeric in (False, True):
        nand = core(numeric=numeric)
        rows = []
        for number, (kind, address, size, at) in enumerate(events):
            result = (write if kind == "w" else read)(nand, f"{kind}{number}", address, size, at)
            rows.append((result.completion_ns, result.pages_read, result.pages_programmed,
                         result.staging_read_hits, result.host_transfer_bytes))
        results.append((rows, nand.page_staging_state()))
    assert results[0] == results[1]


def test_a_batch_takes_staged_pages_off_the_media_path():
    nand = core()
    for index in range(3):
        write(nand, f"row{index}", 10 * PAGE + index * 1024, 1024, 0.)
    batch = nand.submit_many([AccessRequest("kv", Operation.READ, 8 * PAGE, 2 * PAGE + 3 * 1024, 1000.),
                              AccessRequest("other", Operation.READ, 0, 4 * PAGE, 1000.)])
    assert (batch[0].staging_read_hits, batch[0].pages_read) == (1, 2)
    # The staged 3 KiB moves at burst granularity, the media pages at 4 KiB.
    assert batch[0].host_transfer_bytes == 2 * PAGE + 3 * 1024
    assert (batch[1].staging_read_hits, batch[1].pages_read) == (0, 4)


@pytest.mark.parametrize("numeric", [False, True])
def test_external_staging_bills_no_nand_resource_until_the_page_crosses_to_be_programmed(numeric):
    nand = core("external", numeric=numeric, capacity=0)
    rows = [write(nand, f"row{index}", 10 * PAGE + index * 1024, 1024, 0.) for index in range(4)]
    assert [row.completion_ns for row in rows] == [1.] * 4
    assert [row.host_transfer_bytes for row in rows] == [0, 0, 0, PAGE]
    tail = read(nand, "tail", 10 * PAGE, 3 * 1024, 2.)
    assert tail.staging_read_hits == 1
    assert tail.host_transfer_bytes == 0


def test_a_write_with_full_pages_programs_them_directly_and_stages_its_tail():
    nand = core()
    result = write(nand, "import", 20 * PAGE, 3 * PAGE + 512, 0.)
    assert result.pages_programmed == 3
    assert result.counters["staging_write_pages"] == 1
    assert result.physical_read_bytes == 0
    assert nand.page_staging_state()["staged_pages"] == 1


@pytest.mark.parametrize("numeric", [False, True])
def test_a_tail_that_does_not_fit_spills_to_the_array_with_a_host_merge(numeric):
    nand = core(numeric=numeric, capacity=2 * PAGE)
    write(nand, "a", 0 * PAGE, 3000, 0.)       # channel 0
    write(nand, "b", 4 * PAGE, 3000, 0.)       # page 4: channel 4 -> host 0
    assert nand.page_staging_state()["valid_bytes"][0] == 6000
    spilled = write(nand, "c", 8 * PAGE, 3000, 0.)   # page 8: host 0 again, 9000 > 8192
    assert spilled.pages_programmed == 1 and spilled.counters["staging_write_pages"] == 0
    state = nand.page_staging_state()
    assert (state["valid_bytes"][0], state["staged_pages"], state["spilled_pages"]) == (6000, 2, 1)
    # Its later rows stay on the array page: the host reads, merges and writes it.
    more = write(nand, "c2", 8 * PAGE + 3000, 500, 200000.)
    assert more.pages_programmed == 1 and more.physical_read_bytes == PAGE
    back = read(nand, "back", 8 * PAGE, 3500, 900000.)
    assert (back.staging_read_hits, back.pages_read) == (0, 1)
    # A released allocation's pages may be staged again.
    nand.discard_tail_pages(8 * PAGE, PAGE)
    nand.discard_tail_pages(0, PAGE)
    again = write(nand, "d", 8 * PAGE, 3000, 1000000.)
    assert again.pages_programmed == 0 and nand.page_staging_state()["staged_pages"] == 2


@pytest.mark.parametrize("numeric", [False, True])
def test_a_staged_tail_that_outgrows_the_store_is_programmed_and_continues_on_the_array(numeric):
    nand = core(numeric=numeric, capacity=2 * PAGE)
    nand._tail_shared = 3000  # interleaved activations' share per channel
    write(nand, "a", 0, 3000, 0.)
    write(nand, "b", 4 * PAGE, 3000, 0.)       # 6000 staged + 3000 shared: spills
    assert nand.page_staging_state()["staged_pages"] == 1
    write(nand, "a2", 3000, 500, 10.)          # 3500 + 3000 still fits
    nand._tail_shared = 5000                   # the activations grew
    grown = write(nand, "a3", 3500, 100, 20.)
    state = nand.page_staging_state()
    assert (state["evicted_pages"], state["spilled_pages"]) == (1, 2)
    assert grown.pages_programmed >= 1
    # The evicted copy's program runs; the page is read from the array.
    late = read(nand, "late", 0, 3600, 900000.)
    assert (late.staging_read_hits, late.pages_read) == (0, 1)


def test_a_sampled_run_keeps_the_steps_stored_share_of_new_tail_pages():
    nand = core()
    nand._tail_fraction = .25
    for page in range(400):
        write(nand, f"row{page}", page * PAGE + 2048, 512, 0.)
    state = nand.page_staging_state()
    assert 70 <= state["staged_pages"] <= 130 and state["staged_pages"] + state["spilled_pages"] == 400
    # A kept page keeps growing in the store; a spilled one stays on the array.
    kept = min(nand._tail_pages)
    grown = write(nand, "grow", kept * PAGE + 2560, 512, 10.)
    assert grown.counters["staging_write_pages"] == 1 and nand._tail_pages[kept][1] == 3072
    spilled = min(nand._tail_spilled)
    assert write(nand, "rmw", spilled * PAGE + 2560, 512, 10.).physical_read_bytes == PAGE


def test_spilled_pages_roll_back_with_a_transaction():
    nand = core(capacity=PAGE)
    write(nand, "a", 0, 3000, 0.)
    token = nand.begin_transaction()
    write(nand, "b", 4 * PAGE, 3000, 10.)      # spills
    assert nand._tail_spilled == {4}
    nand.rollback_transaction(token)
    assert nand._tail_spilled == set() and nand.page_staging_state()["spilled_pages"] == 0


def test_discard_and_rollback_restore_the_staged_pages():
    nand = core()
    write(nand, "a", 0, 1024, 0.)
    token = nand.begin_transaction()
    write(nand, "b", 1024, 1024, 10.)
    write(nand, "c", 5 * PAGE, 1024, 10.)
    assert nand.page_staging_state()["staged_pages"] == 2
    nand.rollback_transaction(token)
    assert nand._tail_pages[0][1] == 1024 and 5 not in nand._tail_pages
    assert nand.page_staging_state()["valid_bytes"] == [1024, 0, 0, 0]
    assert nand.discard_tail_pages(0, PAGE) == 1
    assert nand.page_staging_state()["staged_pages"] == 0


def test_bound_sram_ports_share_the_stack_channels():
    timeline = ResourceTimeline()
    ids = {direction: [f"nand:host:{stack}:{direction}" for stack in range(4)] for direction in ("read", "write")}
    sram = SramCore(SramConfig(capacity_bytes=1 << 20, burst_bytes=64, interface_bandwidth_gb_s=100.,
        read_bandwidth_gb_s=100., write_bandwidth_gb_s=100., read_latency_ns=1., write_latency_ns=1.,
        ports=4, max_outstanding_requests=16, metadata={"port_resource_ids": ids}), timeline,
        capture_details=False)
    sram.submit(AccessRequest("act", Operation.WRITE, 0, 256, 0.))
    assert all(timeline.busy_ns.get(f"nand:host:{stack}:write") for stack in range(4))
    nand = NandCore(config(), timeline, capture_details=True)
    row = write(nand, "kv", 0, 1024, 0.)
    # The staged row waits for its stack channel behind the activation burst.
    assert row.completion_ns > 1. + 1024 / 25.
    with pytest.raises(ValueError, match="port_resource_ids"):
        SramConfig(capacity_bytes=1 << 20, burst_bytes=64, interface_bandwidth_gb_s=100.,
                   read_bandwidth_gb_s=100., write_bandwidth_gb_s=100., read_latency_ns=1.,
                   write_latency_ns=1., ports=4, max_outstanding_requests=16,
                   metadata={"port_resource_ids": {"read": ids["read"]}})


def test_staging_rejects_other_protocols_and_host_buffers():
    with pytest.raises(ValueError, match="page staging requires"):
        config(write_buffer_bytes=PAGE)
    with pytest.raises(ValueError, match="page_staging must be"):
        config("sram")


@pytest.mark.parametrize("numeric", [False, True])
def test_a_page_opened_mid_way_holds_its_earlier_rows_in_the_store(numeric):
    # A sampled window (or an unpaid seed) never ran the page's earlier
    # appends; they still wait in the store, so the last row fills the page.
    nand = core(numeric=numeric)
    middle = write(nand, "middle", 30 * PAGE + 2048, 1024, 0.)
    assert middle.pages_programmed == 0
    assert nand.page_staging_state()["valid_bytes"][30 % 8 % 4] == 3072
    last = write(nand, "last", 30 * PAGE + 3072, 1024, 10.)
    assert last.pages_programmed == 1 and last.physical_read_bytes == 0
    whole = read(nand, "whole", 30 * PAGE, PAGE, 20.)
    assert (whole.staging_read_hits, whole.pages_read) == (1, 0)


@pytest.mark.parametrize("numeric", [False, True])
def test_a_read_of_never_written_staged_bytes_is_rejected(numeric):
    nand = core(numeric=numeric)
    write(nand, "middle", 30 * PAGE + 2048, 1024, 0.)  # the store holds [0, 3072)
    with pytest.raises(ValueError, match="no write put in the store"):
        read(nand, "hole", 30 * PAGE + 2048, 2048, 10.)
