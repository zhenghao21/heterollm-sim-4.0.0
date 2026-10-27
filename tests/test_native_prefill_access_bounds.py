"""Per-access row envelopes use address arithmetic and an unchanged task DAG."""
from copy import deepcopy

import pytest

from heterollm_sim.native_prefill_access_bounds import _fresh_accesses, per_access_ns
from heterollm_sim.native_row_bounds import _Device, first_access_ns, first_access_penalty_ns
from test_native_row_bounds import devices, _reference


@pytest.mark.parametrize("name", ["hbm4", "ddr", "ddr_row_local", "ddr_dies"])
def test_each_access_matches_a_fresh_burst_mapping_reference(devices, name):
    config = devices[name]
    geometry = _Device(config)
    unit = config.interleave_bytes or config.burst_bytes
    die_bytes = geometry.per_die
    ranges = [(0, 17), (0, 17), (unit - 1, 3 * unit + 2),
              (config.row_bytes - 7, 2 * config.row_bytes + 11),
              (die_bytes - 19, 41)]
    for address, size in ranges:
        if address + size > config.effective_capacity_bytes:
            continue
        expected = _reference(config, address, size, set())
        result = {}
        for path, penalty in _fresh_accesses(geometry, address, size):
            result[path] = result.get(path, 0.0) + penalty
        assert result == pytest.approx(expected)
    # The reusable geometry never acquires an access-history bitmap.
    assert geometry.unseen == len(geometry.seen) and not geometry.seen.any()


def test_repeated_accesses_to_the_same_bank_are_each_charged(devices):
    config = devices["ddr"]
    penalty = first_access_penalty_ns(config)
    facts = [("a", 0.0, 1.0, (), ("command",), (("ddr", 0, 8),)),
             ("b", 1.0, 2.0, (), ("command",), (("ddr", 0, 8),))]
    terms = {"a": "layer", "b": "layer"}
    before = deepcopy(facts)
    assert first_access_ns(facts, {"ddr": config}, terms) == {"layer": penalty}
    assert per_access_ns(facts, {"ddr": config}, terms) == {"layer": 2 * penalty}
    assert facts == before
    # Earlier work outside the term cannot exhaust a future access's bound.
    assert per_access_ns(facts, {"ddr": config}, {"b": "layer"}) == {"layer": penalty}


def test_multiple_contiguous_accesses_sum_on_their_own_command_path(devices):
    config = devices["ddr"]
    penalty = first_access_penalty_ns(config)
    facts = [("a", 0.0, 1.0, (), (), (("ddr", 0, 8), ("ddr", 0, 8)))]
    assert per_access_ns(facts, {"ddr": config}, {"a": "entry"}) == {"entry": 2 * penalty}


def test_parallel_paths_join_by_maximum_and_term_barriers_restart_propagation(devices):
    config = devices["ddr"]
    penalty = first_access_penalty_ns(config)
    access = (("ddr", 0, 8),)
    facts = [("a", 0.0, 1.0, (), ("cpu",), access),
             ("b", 0.0, 1.0, (), ("gpu",), access),
             ("c", 1.0, 2.0, ("a", "b"), (), access),
             ("d", 2.0, 3.0, ("c",), ("cpu",), access)]
    assert per_access_ns(facts, {"ddr": config}, {"a": "x", "b": "x", "c": "x", "d": "y"}) == {
        "x": 2 * penalty, "y": penalty}


def test_non_dram_and_empty_accesses_do_not_create_row_penalties(devices):
    facts = [("a", 0.0, 1.0, (), (), (("nand", 0, 4096), ("ddr", 0, 0))),
             ("b", 1.0, 2.0, ("a",), (), ())]
    assert per_access_ns(facts, {"nand": None, "ddr": devices["ddr"]}, {"a": "x", "b": "x"}) == {"x": 0.0}


def test_large_contiguous_access_uses_bank_vectors_instead_of_every_burst(devices, monkeypatch):
    config = devices["ddr"]
    geometry = _Device(config)
    original = _Device._lanes_banks
    calls = []

    def count(self, address, size):
        calls.append((address, size))
        return original(self, address, size)

    monkeypatch.setattr(_Device, "_lanes_banks", count)
    size = min(1024**3, geometry.per_die)
    result = dict(_fresh_accesses(geometry, 0, size))
    assert len(calls) == 1
    # This range covers every bank of every lane, although it contains many
    # more bursts than there are banks. Each command path gets one penalty
    # per bank; it does not grow with the number of rows in this access.
    assert sum(result.values()) == pytest.approx(
        config.lane_count * config.bank_count * first_access_penalty_ns(config))
