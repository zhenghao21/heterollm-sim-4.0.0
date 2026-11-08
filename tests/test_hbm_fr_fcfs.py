"""The HBM4 per-PC FR-FCFS controller: mapping, legality, policy and path parity."""
from __future__ import annotations

from collections import defaultdict
import random

import numpy as np
import pytest

from heterollm_sim._hbm_control import _FrFcfsController, fr_queue_parameters
from heterollm_sim._hbm_numeric import FR_STAT_FIELDS
from heterollm_sim.dram_core import DramCore, _intervals_add, _intervals_overlap
from heterollm_sim.memory_mapping import dram_request_channels, map_dram_address
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest, DramConfig, Operation
from heterollm_sim.physical_contract import parse_execution_physical_config
from heterollm_sim.physical_presets import resolve_physical_config

READ, WRITE = Operation.READ, Operation.WRITE
GB = 10 ** 9


def test_dependency_intervals_answer_as_the_pairwise_scan_does():
    rng = random.Random(7)
    for _trial in range(200):
        starts, ends, pairs = [], [], []
        for _step in range(40):
            left = rng.randrange(400)
            right = left + rng.randrange(1, 40)
            assert _intervals_overlap(starts, ends, left, right) == any(
                other < right and left < end for other, end in pairs)
            _intervals_add(starts, ends, left, right)
            pairs.append((left, right))
            assert all(end < following for end, following in zip(ends, starts[1:]))
            assert sum(end - start for start, end in zip(starts, ends)) == len(
                {address for start, end in pairs for address in range(start, end)})


def hbm4(**overrides):
    return resolve_physical_config("hbm4_reference", overrides={
        "stacks": 8, "dies_per_stack": 16, "die_density_gbits": 32, "capacity_bytes": 288 * GB,
        "interface_bandwidth_gb_s": 12800., "read_bandwidth_gb_s": 12800., "write_bandwidth_gb_s": 12800.,
        "activate_different_group_ns": 4., "activate_same_group_ns": 6., "four_activate_window_ns": 20.,
        **overrides})


CONFIG = hbm4()
REFAB = hbm4(refresh_policy="all_bank")
TWELVE_HIGH = hbm4(dies_per_stack=12, die_density_gbits=24)


def numeric_core(config=CONFIG):
    core = DramCore(config, ResourceTimeline(), capture_details=False)
    core._hbm_numeric_enabled = True
    return core


def reference_core(config=CONFIG):
    return DramCore(config, ResourceTimeline(), capture_details=True)


def address(lane, *, group=0, bank=0, column=0, sid=0, row=0, config=CONFIG):
    """Invert the SID-aware mapping for one burst."""
    groups_per_sid = config.bank_groups_per_sid
    bursts_per_row = config.row_bytes // config.burst_bytes
    cursor = row * config.sid_count + sid
    cursor = cursor * config.banks_per_group + bank
    cursor = cursor * bursts_per_row + column
    cursor = cursor * config.lane_count + lane
    return (cursor * groups_per_sid + group % groups_per_sid) * config.burst_bytes


def request(name, operation, addr, size=32, arrival=0.):
    return AccessRequest(str(name), operation, addr, size, arrival)


class Recorder(_FrFcfsController):
    """The reference controller with an independent command log for the checker."""

    def __init__(self, core, requests):
        super().__init__(core, requests)
        self.log = []
        self.entry = None

    def issue(self, lane, lanes, kind, start):
        if kind < 3:
            self.entry = lane["queues"][lane["direction"]][lane["offset"]]
        return super().issue(lane, lanes, kind, start)

    def precharge(self, mapping, old, bank, earliest):
        # A row closing on its deadline need not be the served entry's bank.
        served, self.entry = self.entry, dict(mapping=mapping, row=None)
        try:
            return super().precharge(mapping, old, bank, earliest)
        finally:
            self.entry = served

    def refresh_bank(self, key, within, mapping, earliest):
        self.entry = dict(mapping=mapping, row=None)
        return super().refresh_bank(key, within, mapping, earliest)

    def skip_refresh_gap(self, lanes, latest):
        """The checker follows every command."""
        return False

    def command(self, key, kind, earliest, cycles, name, *, rising=False):
        transfer = super().command(key, kind, earliest, cycles, name, rising=rising)
        mapping = self.entry["mapping"] if self.entry is not None else None
        self.log.append(dict(name=name, key=key, start=transfer.start_ns, end=transfer.end_ns,
                             bus=kind, mapping=mapping, row=None if self.entry is None else self.entry["row"]))
        return transfer

    def reserve_clock(self, rid, start, duration, *, direction=None, byte_count=0):
        reservation = super().reserve_clock(rid, start, duration, direction=direction, byte_count=byte_count)
        if byte_count:
            self.log.append(dict(name="DATA", key=rid, start=reservation.start_ns, end=reservation.end_ns,
                                 direction=direction, mapping=self.entry["mapping"]))
        return reservation


def check_legal(log, config=CONFIG):
    """Check every pairwise and window constraint on the recorded commands."""
    ck = config.command_clock_ns
    slack = 1e-6
    buses = defaultdict(list)
    data = defaultdict(list)
    by_pc = defaultdict(list)
    for event in sorted(log, key=lambda item: item["start"]):
        if event["name"] == "DATA":
            data[event["key"]].append(event)
            continue
        buses[(event["key"][:2], event["bus"])].append(event)
        by_pc[event["key"]].append(event)
    for events in list(buses.values()) + list(data.values()):
        for left, right in zip(events, events[1:]):
            assert right["start"] >= left["end"] - slack, (left, right)
    checked = defaultdict(int)
    for events in by_pc.values():
        columns = {READ.value: [], WRITE.value: []}
        write_ends = {}
        activates = []
        banks = {}
        def bank_key(mapping):
            return (mapping.bank_group, mapping.bank)
        last_refresh_end = -1.
        last_bank_refresh = None
        window = []
        refreshed = {}
        set_seen = defaultdict(set)
        set_end = {}
        for event in events:
            name, start, mapping = event["name"], event["start"], event["mapping"]
            if name == "REFRESH_ALL":
                last_refresh_end = start + config.refresh_cycle_ns
                banks = {key: dict(state, open=None, act_allowed=last_refresh_end) for key, state in banks.items()}
                continue
            if name == "REFRESH_CNOP":
                assert any(item["name"] == "REFRESH_ALL" and abs(item["start"] - start) <= slack for item in events), event
                continue
            if name == "PRECHARGE_ALL":
                for state in banks.values():
                    if state.get("open") is not None:
                        assert start >= state["anchor"] + config.minimum_active_ns - slack
                        state["open"] = None
                continue
            if name == "REFRESH_BANK":
                state = banks.setdefault(bank_key(mapping), {"open": None})
                group = mapping.bank_group
                within = group * config.banks_per_group + mapping.bank
                sid = within // 16
                assert state.get("open") is None, event
                assert abs(start / ck - round(start / ck)) < 1e-6, event
                if state.get("precharged") is not None:
                    assert start >= state["precharged"] + config.precharge_to_activate_ns - slack
                if "anchor" in state:
                    assert start >= state["anchor"] + config.activate_cycle_ns - slack
                    checked["REFpb tRC"] += 1
                if "act_allowed" in state:
                    assert start >= state["act_allowed"] - slack
                for other_group, other_anchor in activates[-8:]:
                    gap = config.activate_same_group_ns if other_group == group else config.activate_different_group_ns
                    assert start >= other_anchor + gap - slack
                    checked["REFpb tRRD"] += 1
                if last_bank_refresh is not None:
                    assert start >= last_bank_refresh + config.refresh_bank_gap_ns - slack
                    checked["REFpb tRREFD"] += 1
                window = sorted(window + [start])
                if len(window) >= 5:
                    assert window[-1] >= window[-5] + config.four_activate_window_ns - slack
                    checked["REFpb tFAW"] += 1
                assert within not in set_seen[sid], event
                if not set_seen[sid] and sid in set_end:
                    assert start >= set_end[sid] - slack
                    checked["REFpb set"] += 1
                set_seen[sid].add(within)
                if len(set_seen[sid]) == 16:
                    set_seen[sid].clear()
                    set_end[sid] = start + config.refresh_bank_cycle_ns
                if within in refreshed:
                    assert start - refreshed[within] <= 9 * config.refresh_interval_ns + slack
                refreshed[within] = start
                state["act_allowed"] = start + config.refresh_bank_cycle_ns
                last_bank_refresh = start
                checked["REFpb"] += 1
                continue
            assert start >= last_refresh_end - slack, event
            state = banks.setdefault(bank_key(mapping), {"open": None})
            group = mapping.bank_group
            sid = (group * config.banks_per_group + mapping.bank) // 16
            if name == "ACTIVATE":
                anchor = start + ck
                assert state.get("open") is None
                if "act_allowed" in state:
                    assert anchor >= state["act_allowed"] - slack
                if last_bank_refresh is not None:
                    assert anchor >= last_bank_refresh + config.refresh_bank_gap_ns - slack
                if "anchor" in state:
                    assert anchor >= state["anchor"] + config.activate_cycle_ns - slack
                    checked["tRC"] += 1
                if state.get("precharged") is not None:
                    assert start >= state["precharged"] + config.precharge_to_activate_ns - slack
                    checked["tRP"] += 1
                for other_group, other_anchor in activates[-8:]:
                    gap = config.activate_same_group_ns if other_group == group else config.activate_different_group_ns
                    assert anchor >= other_anchor + gap - slack
                    checked["tRRD_L" if other_group == group else "tRRD_S"] += 1
                window = sorted(window + [anchor])
                if len(window) >= 5:
                    assert window[-1] >= window[-5] + config.four_activate_window_ns - slack
                    checked["tFAW"] += 1
                activates.append((group, anchor))
                state.update(open=event["row"], anchor=anchor, ready_col=anchor + config.activate_to_read_ns)
                state.pop("pre_ready", None)
            elif name == "PRECHARGE":
                assert state.get("open") is not None
                assert start >= state["anchor"] + config.minimum_active_ns - slack
                checked["tRAS"] += 1
                if config.maximum_active_ns > 0:
                    assert start <= state["anchor"] + config.maximum_active_ns + slack
                    checked["tRASmax"] += 1
                if "pre_ready" in state:
                    assert start >= state["pre_ready"] - slack
                    checked["tRTP/tWR"] += 1
                state.update(open=None, precharged=start)
            elif name in ("READ_COMMAND", "WRITE_COMMAND"):
                read = name == "READ_COMMAND"
                op = READ.value if read else WRITE.value
                assert state.get("open") == event["row"], event
                assert start >= state["ready_col"] - slack
                checked["tRCD"] += 1
                for other_start, other_group, other_sid in columns[op][-16:]:
                    gap = (config.column_different_sid_ns if read and other_sid != sid
                           else config.column_same_group_ns if other_group == group
                           else config.column_different_group_ns)
                    assert start >= other_start + gap - slack
                    checked["tCCDR" if read and other_sid != sid else
                            "tCCDL" if other_group == group else "tCCDS"] += 1
                if read:
                    for other_group, end in write_ends.items():
                        gap = (config.write_to_read_same_group_ns if other_group == group
                               else config.write_to_read_different_group_ns)
                        assert start >= end + gap - slack
                        checked["tWTR"] += 1
                    state["pre_ready"] = max(state.get("pre_ready", 0.), start + config.read_to_precharge_ns)
                else:
                    for other_start, _group, _sid in columns[READ.value][-16:]:
                        assert start >= other_start + config.read_to_write_command_ns - slack
                        checked["tRTW"] += 1
                    end = start + config.write_latency_ns + config.burst_bytes / config.effective_lane_bandwidth_gb_s
                    write_ends[group] = end
                    state["pre_ready"] = max(state.get("pre_ready", 0.), end + config.write_to_precharge_ns)
                columns[op].append((start, group, sid))
    return checked


def run_recorded(*batches, config=CONFIG):
    core = reference_core(config)
    results, log = [], []
    for requests in batches:
        recorder = Recorder(core, tuple(requests))
        results += recorder.execute()
        log += recorder.log
    return results, log


def legality_stream():
    """A mixed batch, then writes followed by reads to expose both turnarounds."""
    rng = random.Random(7)

    def burst(index, operation):
        return request(index, operation, address(
            rng.randrange(4), group=rng.randrange(2), bank=rng.randrange(8), column=rng.randrange(32),
            sid=rng.randrange(4), row=rng.randrange(4)))
    return ([burst(index, rng.choice((READ, WRITE))) for index in range(60)],
            [burst(index, WRITE) for index in range(30)],
            [burst(index, READ) for index in range(30)],
            [burst(index, WRITE) for index in range(30)])


def test_command_sequence_covers_and_meets_every_timing_constraint():
    _, log = run_recorded(*legality_stream())
    checked = check_legal(log)
    for name in ("tCCDL", "tCCDS", "tCCDR", "tRRD_L", "tRRD_S", "tFAW", "tRCD", "tRP", "tRAS", "tRC",
                 "tRTP/tWR", "tWTR", "tRTW"):
        assert checked[name] > 0, name


def test_refresh_stream_stays_legal():
    arrival = CONFIG.refresh_interval_ns - 100.
    items = [request(index, READ, address(index % 2, group=index % 2, column=index // 4, bank=(index // 8) % 8),
                     arrival=arrival) for index in range(1500)]
    results, log = run_recorded(items, config=REFAB)
    assert any(event["name"] == "REFRESH_ALL" for event in log)
    check_legal(log, REFAB)
    assert max(result.completion_ns for result in results) > CONFIG.refresh_interval_ns


def dense_stream(config, count=2500, arrival=0.):
    """Row misses and hits over every SID of two PCs in one channel."""
    rng = random.Random(5)
    sids = config.sid_count
    return [request(index, rng.choice((READ, READ, WRITE)), address(
        rng.randrange(2), group=rng.randrange(2), bank=rng.randrange(8), column=rng.randrange(32),
        sid=rng.randrange(sids), row=rng.randrange(3), config=config), arrival=arrival) for index in range(count)]


@pytest.mark.parametrize("config", (CONFIG, TWELVE_HIGH), ids=("16H", "12H"))
def test_per_bank_refresh_under_traffic_meets_every_constraint(config):
    results, log = run_recorded(dense_stream(config), config=config)
    checked = check_legal(log, config)
    for name in ("REFpb", "REFpb tRC", "REFpb tRRD", "REFpb tRREFD", "REFpb tFAW", "tFAW", "tRRD_L", "tRRD_S"):
        assert checked[name] > 0, name
    span = max(result.completion_ns for result in results)
    step = config.refresh_interval_ns / config.bank_count
    # Each PC refreshes in step, at most one tREFI late.
    for pseudo in range(2):
        starts = [event["start"] for event in log if event["name"] == "REFRESH_BANK" and event["key"][2] == pseudo]
        assert len(starts) >= (span - 2 * config.refresh_interval_ns) // step
        for number, start in enumerate(starts):
            assert start - (config.refresh_interval_ns + number * step) <= 1.1 * config.refresh_interval_ns
    assert not any(event["name"] == "REFRESH_ALL" for event in log)


@pytest.mark.parametrize("dies,density", ((4, 24), (8, 24), (12, 24), (16, 32)))
def test_a_gap_catches_up_every_refresh_in_every_organization(dies, density):
    config = hbm4(dies_per_stack=dies, die_density_gbits=density, capacity_bytes=64 * GB)
    first = [request("a", READ, address(0, config=config)), request("b", READ, address(1, config=config))]
    gap = 3 * config.refresh_interval_ns
    second = [request("c", READ, address(0, row=1, config=config), arrival=gap)]
    _, log = run_recorded(first, second, config=config)
    checked = check_legal(log, config)
    step = config.refresh_interval_ns / config.bank_count
    # The epoch opens at one tREFI; the last batch adds the REFpb due while it runs.
    expected = 2 * round((gap - config.refresh_interval_ns) / step)
    assert expected <= sum(event["name"] == "REFRESH_BANK" for event in log) <= expected + 2 * round(400. / step)
    assert checked["REFpb"] >= expected


@pytest.mark.parametrize("config", (CONFIG, TWELVE_HIGH), ids=("16H", "12H"))
def test_skipping_a_gap_changes_nothing_later_commands_can_see(config, monkeypatch):
    def run(skip):
        if not skip:
            monkeypatch.setattr(_FrFcfsController, "skip_refresh_gap", lambda self, lanes, latest: False)
        core = reference_core(config)
        results = []
        interval = config.refresh_interval_ns
        for arrival in (0., 1.5 * interval, 1.5 * interval + 300., 2.1 * interval, 40 * interval, 40 * interval + 700.):
            results += core.submit_many(dense_stream(config, 200, arrival))
        monkeypatch.undo()
        timeline = core.timeline
        return ([result.completion_ns for result in results], sorted(timeline.ready_ns.items()),
                sorted(timeline.bytes_moved.items()), sorted((key, bank.open_row) for key, bank in core._banks.items()),
                {key: (state["refresh_gate"], state["refresh_group_gate"], state["next_refresh"],
                        state["activate_window"])
                 for key, state in core._hbm_control_state["pcs"].items()},
                sum(result.counters.get("refresh_commands", 0) for result in results),
                sum(result.counters.get("precharge_commands", 0) for result in results)), dict(timeline.busy_ns)
    skipped, skipped_busy = run(True)
    stepped, stepped_busy = run(False)
    assert skipped == stepped
    assert skipped_busy == pytest.approx(stepped_busy, rel=1e-9)


def test_a_gap_after_traffic_skips_refresh_that_cannot_be_seen():
    config = TWELVE_HIGH
    core = reference_core(config)
    calls = []

    class Counting(_FrFcfsController):
        def refresh_bank(self, key, within, mapping, earliest):
            calls.append(key)
            return super().refresh_bank(key, within, mapping, earliest)

    Counting(core, tuple(dense_stream(config, 100))).execute()
    calls.clear()
    results = Counting(core, tuple(dense_stream(config, 100, 20 * config.refresh_interval_ns))).execute()
    total = sum(result.counters.get("refresh_commands", 0) for result in results)
    assert total >= 2 * 18 * config.bank_count
    # Only the REFpb a later command could see, and those the batch itself runs over, are simulated.
    assert len(calls) < total / 4


def test_a_refreshing_bank_holds_up_only_its_own_entries():
    step = CONFIG.refresh_interval_ns / CONFIG.bank_count
    due = CONFIG.refresh_interval_ns + 3 * step
    # A row on the bank due next is opened and refreshed, then its next row must wait tRFCpb.
    refreshed = address(0, bank=3)
    first = [request("open", READ, refreshed, arrival=due - 60.)]
    blocked = request("blocked", READ, address(0, bank=3, row=1), arrival=due + 20.)
    others = [request(f"other{index}", READ, address(0, bank=index + 4, sid=1), arrival=due + 20.)
              for index in range(3)]
    results, log = run_recorded(first, [blocked, *others])
    completions = {result.request_id: result.completion_ns for result in results}
    assert all(completions[item.request_id] < completions["blocked"] for item in others)
    refresh = next(event["start"] for event in log if event["name"] == "REFRESH_BANK"
                   and event["mapping"].bank == 3 and event["mapping"].bank_group == 0)
    assert completions["blocked"] >= refresh + CONFIG.refresh_bank_cycle_ns
    check_legal(log)


def test_queued_hits_keep_a_row_open_past_the_refresh_it_is_due():
    step = CONFIG.refresh_interval_ns / CONFIG.bank_count
    due = CONFIG.refresh_interval_ns + 3 * step
    # A run of hits on the bank whose REFpb falls due in the middle of it, then one more entry to end after it.
    items = [request(index, READ, address(0, bank=3, column=index), arrival=due - 15.) for index in range(30)]
    items.append(request("last", READ, address(0, bank=4, sid=1), arrival=due + 400.))
    results, log = run_recorded(items)
    assert sum(result.counters["row_hits"] for result in results) == 29
    refresh = next(event["start"] for event in log if event["name"] == "REFRESH_BANK"
                   and event["mapping"].bank == 3 and event["mapping"].bank_group == 0 and event["key"][2] == 0)
    columns = [event["start"] for event in log if event["name"] == "READ_COMMAND" and event["mapping"].bank == 3]
    assert refresh > max(columns) > due
    check_legal(log)


def test_refresh_commands_are_counted_on_both_paths():
    compiled, scalar = numeric_core(), reference_core()
    items = dense_stream(CONFIG, 600, CONFIG.refresh_interval_ns)
    left = compiled.submit_many(items)
    right = scalar.submit_many(items)
    counts = [sum(result.counters.get("refresh_commands", 0) for result in results) for results in (left, right)]
    assert counts[0] == counts[1] > 0


def test_mapping_keeps_one_lane_row_stream_with_alternating_groups():
    mapped = [map_dram_address(CONFIG, burst * 2 * CONFIG.lane_count * 32) for burst in range(32)]
    assert {item.lane for item in mapped} == {0}
    assert [item.column for item in mapped] == list(range(32))
    assert {(item.bank_group, item.bank, item.row) for item in mapped} == {(0, 0, 0)}
    other = map_dram_address(CONFIG, 32)
    assert (other.lane, other.bank_group, other.column) == (0, 1, 0)
    assert map_dram_address(CONFIG, 64).lane == 1


def test_request_channels_follow_the_two_burst_stripe():
    assert dram_request_channels(CONFIG, request("x", READ, 0, 64)) == dram_request_channels(
        CONFIG, request("y", READ, 32, 32))
    assert len(dram_request_channels(CONFIG, request("z", READ, 0, 128))) == 1
    assert len(dram_request_channels(CONFIG, request("w", READ, 0, 256))) == 2


def fingerprint(core, results):
    timeline = core.timeline
    return ([(result.request_id, result.completion_ns, result.logical_bytes, result.transfer_bytes,
              *[result.counters.get(name) for name in (
                  "row_hits", "row_misses", "row_conflicts", "burst_count", "queue_wait_ns",
                  "physical_read_bytes", "physical_write_bytes")]) for result in results],
            sorted(timeline.ready_ns.items()), sorted(timeline.busy_ns.items()),
            sorted(timeline.bytes_moved.items()), sorted(timeline.directions.items()),
            sorted((key, bank.open_row, bank.ready_ns, bank.activation_ready_ns)
                   for key, bank in core._banks.items()),
            core.hbm_scheduler_stats().tolist())


def batches_of(seed):
    rng = random.Random(seed)
    now = 0.
    batches = []
    for index in range(6):
        items = []
        for position in range(rng.randint(1, 4)):
            items.append(request(f"{index}:{position}", rng.choice((READ, WRITE)),
                                 rng.randrange(0, 1 << 26) // 32 * 32,
                                 rng.choice((32, 64, 96, 4096, 40000)), now))
        batches.append(items)
        now += rng.choice((0., 50., 400., 5000.))
    return batches


@pytest.mark.parametrize("config", (CONFIG, TWELVE_HIGH, REFAB), ids=("16H", "12H", "REFab"))
@pytest.mark.parametrize("seed", range(6))
def test_scalar_and_compiled_paths_are_bit_identical(seed, config):
    compiled, scalar = numeric_core(config), reference_core(config)
    first, second = [], []
    for items in batches_of(seed):
        first += compiled.submit_many(items)
        second += scalar.submit_many(items)
    assert fingerprint(compiled, first) == fingerprint(scalar, second)


@pytest.mark.parametrize("config", (CONFIG, TWELVE_HIGH), ids=("16H", "12H"))
def test_dense_traffic_with_refresh_is_bit_identical_on_both_paths(config):
    compiled, scalar = numeric_core(config), reference_core(config)
    first, second = [], []
    for arrival in (0., 9000., 9100., 11000., 60000., 500000.):
        items = dense_stream(config, 500, arrival)
        first += compiled.submit_many(items)
        second += scalar.submit_many(items)
    assert fingerprint(compiled, first) == fingerprint(scalar, second)
    left, right = compiled._hbm_control_state, scalar._hbm_control_state
    assert left["pcs"].keys() == right["pcs"].keys()
    for key, state in right["pcs"].items():
        for name in ("last_precharge", "blocked_until", "refresh_gate", "refresh_group_gate",
                     "next_refresh", "scheduler_stats"):
            assert left["pcs"][key][name] == state[name], (key, name)


def test_controller_state_publishes_identically_on_both_paths():
    compiled, scalar = numeric_core(), reference_core()
    for items in batches_of(11):
        compiled.submit_many(items)
        scalar.submit_many(items)
    left, right = compiled._hbm_control_state, scalar._hbm_control_state
    assert left["pcs"].keys() == right["pcs"].keys()
    for key, state in right["pcs"].items():
        for name in ("activate", "column", "write_end", "last_precharge", "blocked_until",
                     "next_data_start", "activate_window", "scheduler_stats", "refresh_gate",
                     "refresh_group_gate", "next_refresh"):
            assert left["pcs"][key][name] == state[name], (key, name)
    assert left["channels"] == right["channels"]


def test_rollback_restores_counters_and_controller():
    plain, rolled = numeric_core(), numeric_core()
    head, *rest = batches_of(5)
    plain.submit_many(head)
    rolled.submit_many(head)
    token = rolled.begin_transaction()
    for items in rest[:2]:
        rolled.submit_many(items)
    rolled.rollback_transaction(token)
    assert rolled.hbm_scheduler_stats().tolist() == plain.hbm_scheduler_stats().tolist()
    assert {key: bank.open_row for key, bank in rolled._banks.items()
            if bank.open_row is not None} == {key: bank.open_row for key, bank in plain._banks.items()
                                              if bank.open_row is not None}


def test_streaming_read_stays_under_the_ceiling_with_row_hits():
    size = 8 << 20
    core = numeric_core()
    result = core.submit(request("stream", READ, 0, size, 100.))
    bursts = size // 32
    assert result.counters["row_hits"] >= 0.96 * bursts
    bandwidth = size / (result.completion_ns - 100.)
    assert 9000. < bandwidth <= 12800.
    assert result.counters["read_commands"] == bursts


def test_younger_row_hit_goes_ahead_of_an_older_conflict():
    open_row = request("open", READ, address(0, bank=1, row=2))
    hit = request("hit", READ, address(0, bank=1, row=2, column=5))
    conflict = request("conflict", READ, address(0, bank=1, row=3))
    core = numeric_core()
    core.submit(open_row)
    conflict_result, hit_result = core.submit_many((conflict, hit))
    assert hit_result.counters["row_hits"] == 1
    assert conflict_result.counters["row_conflicts"] == 1
    assert hit_result.completion_ns < conflict_result.completion_ns


def test_conflicting_row_closes_ahead_of_its_turn():
    """Lookahead overlaps a later bank's tRP and tRCD with other banks' traffic."""
    core = numeric_core()
    stream = [request(i, READ, address(0, bank=0, column=i)) for i in range(16)]
    later = request("later", READ, address(0, bank=3, row=5))
    other = numeric_core()
    other.submit(request("open", READ, address(0, bank=3, row=4)))
    core.submit(request("open", READ, address(0, bank=3, row=4)))
    results = core.submit_many((*stream, later))
    assert results[-1].counters["row_conflicts"] == 1
    # PRE, tRP, ACT and tRCD of the later bank fit inside the stream's reads.
    stream_end = max(result.completion_ns for result in results[:-1])
    assert results[-1].completion_ns <= stream_end + 2 * CONFIG.column_same_group_ns


def test_read_goes_first_until_the_write_queue_reaches_its_watermark():
    read_depth, write_depth, high, low, min_reads = fr_queue_parameters(CONFIG)
    assert (read_depth, write_depth, high, low, min_reads) == (32, 64, 55, 32, 16)
    core = numeric_core()
    writes = [request(f"w{i}", WRITE, address(0, column=i % 32, row=1 + i // 32)) for i in range(40)]
    read = request("r", READ, address(0, bank=2))
    results = core.submit_many((*writes, read))
    assert results[-1].completion_ns < min(result.completion_ns for result in results[:-1])
    assert core.hbm_scheduler_stats()[0, FR_STAT_FIELDS.index("write_drains")] == 0


def test_write_drain_starts_and_stops_on_its_watermarks():
    core = numeric_core()
    reads = [request(f"r{i}", READ, address(0, bank=1, column=i % 32, row=i // 32)) for i in range(400)]
    writes = [request(f"w{i}", WRITE, address(0, bank=2, column=i % 32, row=i // 32)) for i in range(400)]
    core.submit_many((*reads, *writes))
    stats = core.hbm_scheduler_stats()[0]
    assert stats[FR_STAT_FIELDS.index("write_drains")] >= 2
    assert stats[FR_STAT_FIELDS.index("peak_read_queue")] == 32
    assert stats[FR_STAT_FIELDS.index("peak_write_queue")] == 64
    assert stats[FR_STAT_FIELDS.index("direction_switches")] >= 4


def test_read_after_overlapping_write_waits_for_its_completion():
    target = address(0, bank=1)
    core = numeric_core()
    write, read = core.submit_many((request("w", WRITE, target), request("r", READ, target)))
    assert read.completion_ns > write.completion_ns + CONFIG.read_latency_ns
    assert read.arrival_ns == 0.
    assert read.counters["queue_wait_ns"] >= write.completion_ns


def test_other_channels_are_not_held_back_by_an_earlier_later_arrival():
    """Per-PC queues keep no device-wide admission order."""
    core = numeric_core()
    core.submit(request("late", READ, address(0), 32, 5000.))
    early = core.submit(request("early", READ, address(4), 32, 10.))
    assert early.completion_ns < 5000.


def test_results_keep_input_order_and_wait_counts_only_other_traffic():
    core = numeric_core()
    stream = request("first", READ, 0, 1 << 20, 0.)
    alone = numeric_core().submit(request("first", READ, 0, 1 << 20, 0.))
    assert alone.counters["queue_wait_ns"] == 0.
    first = core.submit(stream)
    second = core.submit(request("second", READ, 1 << 20, 1 << 20, 0.))
    assert second.counters["queue_wait_ns"] > 0.
    assert second.counters["queue_wait_ns"] <= first.completion_ns


def test_mapping_scheduler_and_queue_scope_travel_together():
    payload = dict(CONFIG.__dict__)
    with pytest.raises(ValueError, match="go together"):
        DramConfig(**{**payload, "scheduler": "fcfs"})
    with pytest.raises(ValueError, match="go together"):
        DramConfig(**{**payload, "queue_scope": "device"})


def test_batch_resource_bill_equals_the_sum_of_its_public_resources():
    from heterollm_sim._hbm_pack import HBMResourceAccumulator
    accumulated, public = numeric_core(), numeric_core()
    items = [request(i, READ, i * (1 << 16), 1 << 16, 0.) for i in range(4)]
    bill = HBMResourceAccumulator()
    accumulated.submit_many(items, _resource_accumulator=bill)
    results = public.submit_many(items)
    totals = defaultdict(float)
    for result in results:
        for rid, value in result.counters.get("resource_busy_ns", {}).items():
            totals[rid] += value
    assert dict(bill.resource_busy_ns) == dict(totals)
    assert accumulated.timeline.bytes_moved == public.timeline.bytes_moved
    assert sum(public.timeline.bytes_moved.get(rid, 0) for rid in public.timeline.bytes_moved) == 4 << 16


def mirror_batches(seed, config):
    """Whole-stripe traffic on every channel, mixed with small requests that age some channels' state."""
    rng = random.Random(seed)
    lanes = config.lane_count
    stripe = 2 * lanes
    arrival = 1000.
    for _ in range(30):
        arrival += rng.choice((0., 50., 400., 2500., 5000., 42000.))
        count = rng.randint(1, 3)
        firsts, stops, reads = [], [], []
        for _ in range(count):
            first = rng.randrange(1 << 18) * stripe + rng.choice((0, 0, 2 * rng.randrange(lanes), rng.randrange(stripe)))
            firsts.append(first)
            stops.append(first + rng.choice((64, 128, 300, 700)) * stripe + rng.choice((0, 0, rng.randrange(stripe))))
            reads.append(rng.random() < .8)
        if rng.random() < .5:
            lane = rng.randrange(lanes)
            firsts.append(rng.randrange(1 << 18) * stripe + 2 * lane)
            stops.append(firsts[-1] + rng.randint(1, 40))
            reads.append(rng.random() < .5)
        yield (np.array(firsts, dtype=np.int64), np.array(stops, dtype=np.int64),
               np.full(len(firsts), arrival, dtype=np.float64), np.array(reads, dtype=np.bool_))


@pytest.mark.parametrize("seed", range(3))
def test_fr_fcfs_outcome_does_not_depend_on_the_worker_count(seed, monkeypatch):
    from heterollm_sim import _hbm_numeric
    from heterollm_sim._compiled_hbm import _pack_hbm_numeric

    config = hbm4()
    seen = []
    for workers in (1, 2, 4, 8):
        monkeypatch.setattr(_hbm_numeric, "_CHANNEL_WORKERS", workers)
        packed = _pack_hbm_numeric(numeric_core(config), request("x", READ, 0))
        arrays = [array.copy() for array in packed.arrays]
        outcome = []
        for firsts, stops, arrivals, reads in mirror_batches(seed, config):
            try:
                results = _hbm_numeric.run_fr_fcfs(firsts, stops, arrivals, reads, packed._fr_queue,
                                                   packed.g, packed.t, *arrays)
            except ValueError:
                # Random traffic can hold a row past tRASmax.
                outcome.append("refused")
                break
            outcome.append(tuple(value.tobytes() for value in results))
        windows, counts = arrays[12], arrays[13]
        outcome.append(tuple(array.tobytes() for index, array in enumerate(arrays) if index != 12))
        outcome.append([row[:count].tobytes() for row, count in zip(windows, counts)])
        seen.append(outcome)
    assert all(outcome == seen[0] for outcome in seen)


def test_kernel_reports_its_failures_as_python_errors():
    """The kernel runs without the GIL; its failures must still raise, with the controller's messages."""
    from heterollm_sim._compiled_hbm import _pack_hbm_numeric
    from heterollm_sim._hbm_numeric import _run_fr_fcfs_channels

    packed = _pack_hbm_numeric(numeric_core(), request("x", READ, 0))
    half = float(packed.t[0]) / 2

    def run(arrival=0., t=packed.t, prepare=lambda arrays: None):
        arrays = [array.copy() for array in packed.arrays]
        prepare(arrays)
        _run_fr_fcfs_channels(np.array([0], dtype=np.int64), np.array([2], dtype=np.int64), np.array([arrival]),
                              np.array([True]), packed._fr_queue, packed.g, t, *arrays,
                              np.zeros((1, 3), dtype=np.int64), np.zeros((1, 2)), 0, 1, 1, np.zeros((1, 7)))

    run()
    with pytest.raises(OverflowError, match="clock index"):
        run(arrival=half * 2. ** 53)

    def full_calendar(arrays):
        arrays[12][0] = 1e9 + np.arange(arrays[12].shape[1])
        arrays[13][0] = arrays[12].shape[1]
    with pytest.raises(OverflowError, match="ACT calendar"):
        run(prepare=full_calendar)
    short = packed.t.copy()
    short[29] = 1.
    with pytest.raises(ValueError, match="tRASmax activation window"):
        run(t=short)
    wrong_layout = [array.copy() for array in packed.arrays]
    wrong_layout[1] = wrong_layout[1][:, :6].copy()
    with pytest.raises(ValueError, match="layout"):
        _run_fr_fcfs_channels(np.array([0], dtype=np.int64), np.array([2], dtype=np.int64), np.array([0.]),
                              np.array([True]), packed._fr_queue, packed.g, packed.t, *wrong_layout,
                              np.zeros((1, 3), dtype=np.int64), np.zeros((1, 2)), 0, 1, 1, np.zeros((1, 7)))


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("policy,overrides", (
    ("all_bank", {}), ("disabled", {}), ("per_bank", {}), ("per_bank", {"dies_per_stack": 12, "die_density_gbits": 24})))
def test_channel_mirror_matches_simulating_every_channel(seed, policy, overrides):
    from heterollm_sim._compiled_hbm import _pack_hbm_numeric
    from heterollm_sim._hbm_numeric import _run_fr_fcfs_channels

    config = hbm4(refresh_policy=policy, **overrides)
    packed = _pack_hbm_numeric(numeric_core(config), request("x", READ, 0, ))
    channels = config.lane_count // 2
    whole = [array.copy() for array in packed.arrays]
    single = [array.copy() for array in packed.arrays]
    for firsts, stops, arrivals, reads in mirror_batches(seed, config):
        common = (firsts, stops, arrivals, reads, packed._fr_queue, packed.g, packed.t)
        outcome = np.zeros((len(firsts), 3), dtype=np.int64), np.zeros((len(firsts), 2), dtype=np.float64)
        outcome[1][:, 0] = arrivals
        expected = np.zeros((len(firsts), 3), dtype=np.int64), np.zeros((len(firsts), 2), dtype=np.float64)
        expected[1][:, 0] = arrivals
        counts = np.zeros(7)
        try:
            for channel in range(channels):
                row = np.zeros((1, 7))
                _run_fr_fcfs_channels(*common, *single, *expected, channel, channel + 1, 1, row)
                counts += row[0]
        except ValueError as error:
            # Random traffic can hold a row past tRASmax; mirroring must refuse it too.
            with pytest.raises(ValueError, match=str(error)):
                _run_fr_fcfs_channels(*common, *whole, *outcome, 0, channels, channels, np.zeros((channels, 7)))
            return
        rows = np.zeros((channels, 7))
        _run_fr_fcfs_channels(*common, *whole, *outcome, 0, channels, channels, rows)
        mirrored = np.zeros(7)
        for row in rows:
            mirrored += row
        assert outcome[0].tobytes() == expected[0].tobytes()
        assert outcome[1].tobytes() == expected[1].tobytes()
        assert mirrored[:5].tobytes() == counts[:5].tobytes()
        assert mirrored[5:] == pytest.approx(counts[5:], rel=1e-9)
        for name, left, right in zip(("rows", "bf", "addresses", "pc", "history", "pc_seen", "channel", "channel_int",
                                      "resources", "moved", "directions", "touched", "windows", "window_counts", "stats"),
                                     whole, single):
            if name == "resources":
                # Busy totals add per channel when mirroring: the same sums, in another order.
                assert left[:, 1] == pytest.approx(right[:, 1], rel=1e-9)
                left, right = np.delete(left, 1, axis=1), np.delete(right, 1, axis=1)
            elif name == "windows":
                # Slots past an ACT window's count hold nothing.
                left = [row[:count].tobytes() for row, count in zip(left, whole[13])]
                right = [row[:count].tobytes() for row, count in zip(right, single[13])]
                assert left == right, name
                continue
            assert left.tobytes() == right.tobytes(), name


ORGANIZATIONS = (("hbm4_reference", "HBM4", 32, 12, 1600.), ("hbm4_reference", "HBM4", 32, 8, 1600.),
                 ("hbm4_reference", "HBM4", 32, 4, 1600.), ("hbm4_reference", "HBM4", 24, 12, 1600.),
                 ("hbm3e_reference", "HBM3E", 32, 16, 1228.8), ("hbm3e_reference", "HBM3E", 24, 12, 1228.8),
                 ("hbm3e_reference", "HBM3E", 16, 16, 1228.8), ("hbm3e_reference", "HBM3E", 16, 4, 1228.8),
                 ("hbm3e_reference", "HBM3", 16, 8, 819.2), ("hbm3e_reference", "HBM2E", 16, 8, 460.8),
                 ("hbm3e_reference", "HBM2", 16, 4, 307.2))


def organization(profile, generation, density, height, per_stack):
    """Eight stacks of one JEDEC organization at its own interface bandwidth."""
    return resolve_physical_config(profile, overrides={
        "generation": generation, "stacks": 8, "dies_per_stack": height, "die_density_gbits": density,
        "capacity_bytes": density * height * (1 << 30), "interface_bandwidth_gb_s": per_stack * 8,
        "read_bandwidth_gb_s": per_stack * 8, "write_bandwidth_gb_s": per_stack * 8,
        "activate_different_group_ns": 4., "activate_same_group_ns": 6., "four_activate_window_ns": 20.})


@pytest.mark.parametrize("row", ORGANIZATIONS, ids=lambda row: f"{row[1]}-{row[3]}H-{row[2]}Gb")
def test_every_organization_reaches_its_banks_stays_legal_and_keeps_path_parity(row):
    config = organization(*row)
    rng = random.Random(1)
    locations = [map_dram_address(config, rng.randrange(config.effective_capacity_bytes // 32) * 32)
                 for _ in range(5000)]
    assert len({(item.bank_group, item.bank) for item in locations}) == config.bank_count
    assert map_dram_address(config, config.effective_capacity_bytes - 1).row == config.rows_per_bank - 1

    numeric, scalar = numeric_core(config), reference_core(config)
    fast, slow, now = [], [], 0.
    for index in range(6):
        items = [request(f"{index}:{position}", rng.choice((READ, WRITE)),
                         rng.randrange(config.effective_capacity_bytes - (1 << 20)),
                         rng.choice((32, 64, 96, 4096, 40000)), now) for position in range(rng.randint(1, 4))]
        fast += numeric.submit_many(items)
        slow += scalar.submit_many(items)
        now += rng.choice((0., 50., 400., 5000.))
    assert fingerprint(numeric, fast) == fingerprint(scalar, slow)

    def burst(index, operation):
        return request(index, operation, address(
            rng.randrange(4), group=rng.randrange(config.bank_groups_per_sid), bank=rng.randrange(config.banks_per_group),
            column=rng.randrange(32), sid=rng.randrange(config.sid_count), row=rng.randrange(4), config=config))
    _, log = run_recorded([burst(index, rng.choice((READ, WRITE))) for index in range(200)],
                          [burst(index, WRITE) for index in range(60)], [burst(index, READ) for index in range(60)],
                          config=config)
    check_legal(log, config)

    size = 16 << 20
    stream = numeric_core(config).submit(request("stream", READ, 0, size))
    assert .75 * config.interface_bandwidth_gb_s < size / stream.completion_ns <= config.interface_bandwidth_gb_s


def test_refresh_free_organization_meets_row_deadlines_on_both_paths():
    config = organization("hbm3e_reference", "HBM3E", 24, 12, 1228.8)
    assert config.refresh_policy == "disabled" and config.maximum_active_ns > 0
    rng = random.Random(3)

    def burst(name, arrival):
        return request(name, READ, address(
            rng.randrange(2), group=rng.randrange(config.bank_groups_per_sid), bank=rng.randrange(config.banks_per_group),
            column=rng.randrange(32), sid=rng.randrange(config.sid_count), row=rng.randrange(2), config=config),
            arrival=arrival)
    # Open rows first, then land new commands on their deadlines.
    batches = [[burst(f"open:{index}", 0.) for index in range(48)]]
    batches += [[burst(f"{step}:{index}", config.maximum_active_ns + 5. * step) for index in range(12)]
                for step in range(-4, 40)]
    numeric, scalar = numeric_core(config), reference_core(config)
    fast, slow = [], []
    for items in batches:
        fast += numeric.submit_many(items)
        slow += scalar.submit_many(items)
    assert fingerprint(numeric, fast) == fingerprint(scalar, slow)
    results, log = run_recorded(*batches, config=config)
    assert check_legal(log, config)["tRASmax"] > 0
    assert sum(result.counters["precharge_commands"] for result in results) > 0
