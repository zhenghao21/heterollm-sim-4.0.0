"""Boundary row-state bounds of native stationary sample windows."""
from dataclasses import replace
import random
import sys
from pathlib import Path

import pytest

from heterollm_sim import native_row_bounds as rows
from heterollm_sim.memory_mapping import split_dram_request
from heterollm_sim.memory_types import AccessRequest, Operation
from heterollm_sim.physical_contract import parse_execution_physical_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(scope="module")
def devices():
    from exploring_hbf_native_config import build_native_comparison_scenario

    scenario = build_native_comparison_scenario("llama3_405b", "HBM4", "SHORT", batch_size=8, tp=8)
    configs = {component.component_id: parse_execution_physical_config(component.metadata["physical_memory_config"])
               for component in scenario.hardware.components if component.component_id in ("hbm0", "hostmem0")}
    host = configs["hostmem0"]
    return {"hbm4": configs["hbm0"], "ddr": host, "ddr_row_local": replace(host, address_mapping="row_local", interleave_bytes=None),
            # Accesses here cross die and stack boundaries.
            "ddr_dies": replace(host, rows_per_bank=4, dies_per_stack=2, stacks=2, capacity_bytes=None)}


def _reference(config, address, byte_count, seen):
    """First-touched banks of one access, burst by burst through map_dram_address."""
    lanes, banks = config.lane_count, config.bank_count
    per_die = lanes * banks * config.rows_per_bank * config.row_bytes
    per_stack = per_die * config.addressable_die_count
    sums = {}
    for segment in split_dram_request(AccessRequest("r", Operation.READ, address, byte_count, 0.), config):
        mapping = segment.mapping
        bank = ((mapping.rank * config.bank_groups_per_rank + mapping.bank_group) * config.banks_per_group
                + mapping.bank)
        stack = segment.address // per_stack
        key = ((stack * config.addressable_die_count + mapping.die) * lanes + mapping.lane) * banks + bank
        if key in seen:
            continue
        seen.add(key)
        path = (mapping.stack * config.channels + mapping.channel if config.control_mode == "jedec_hbm"
                else mapping.lane)
        sums[path] = sums.get(path, 0.) + rows.first_access_penalty_ns(config)
    return sums


@pytest.mark.parametrize("name", ["hbm4", "ddr", "ddr_row_local", "ddr_dies"])
def test_first_touched_banks_and_command_paths_match_the_address_mapping(devices, name):
    config = devices[name]
    rng = random.Random(name)
    device, seen = rows._Device(config), set()
    span = config.lane_count * config.bank_count * (config.interleave_bytes or config.burst_bytes)
    largest = 2 * span + 4 * config.row_bytes
    limit = min(64 * span, config.effective_capacity_bytes - largest)
    per_die = config.lane_count * config.bank_count * config.rows_per_bank * config.row_bytes
    # Ranges that straddle every die and stack boundary inside the device.
    straddles = [(boundary - config.row_bytes, 2 * config.row_bytes + 5)
                 for boundary in range(per_die, config.effective_capacity_bytes, per_die)]
    randoms = [(rng.randrange(0, limit), rng.choice([1, 31, config.burst_bytes, 3 * config.row_bytes + 17,
                                                    rng.randrange(1, 2 * span)])) for _ in range(60)]
    for address, byte_count in [*randoms[:30], *straddles, *randoms[30:]]:
        expected = _reference(config, address, byte_count, seen)
        got = {}
        for path, penalty in device.first_accesses(address, byte_count):
            got[path] = got.get(path, 0.) + penalty
        assert got == pytest.approx(expected)
    assert device.unseen == config.lane_count * config.bank_count * config.addressable_die_count \
        * config.addressable_stack_count - len(seen)


def test_penalty_spans_a_ready_hit_to_a_conflict_behind_a_fresh_activation(devices):
    hbm, ddr = devices["hbm4"], devices["ddr"]
    # tRC 48.64 >= tRAS+tRP 48.64, tRCD 14, tFAW 20, tPPD 1.28, (1.5+.5+2) CK of .64.
    assert rows.first_access_penalty_ns(hbm) == pytest.approx(48.64 + 14. + 20. + 1.28 + 4. * .64)
    # Write recovery 15, PRE 14, ACT 14; no bank-group gates.
    assert rows.first_access_penalty_ns(ddr) == pytest.approx(15. + 14. + 14.)


def test_term_delay_adds_along_dependencies_and_resource_order_only_inside_a_term(devices):
    ddr = devices["ddr"]
    penalty = rows.first_access_penalty_ns(ddr)
    bank = ddr.interleave_bytes
    facts = [
        # a and c reach two banks first; b follows a on its resource; d
        # depends on a and c; e is in the next term.
        ("a", 0., 10., (), ("cpu",), (("ddr", 0, 8),)),
        ("c", 0., 10., (), ("gpu",), (("ddr", bank, 8),)),
        ("b", 10., 20., (), ("cpu",), (("ddr", 0, 8),)),
        ("d", 20., 30., ("a", "c"), ("x",), (("ddr", 2 * bank, 8),)),
        ("e", 30., 40., ("d",), ("cpu",), (("ddr", 3 * bank, 8),)),
    ]
    terms = {"a": "t", "b": "t", "c": "t", "d": "t", "e": "u"}
    result = rows.first_access_ns(facts, {"ddr": ddr}, terms)
    # One path (shared command lane): a -> d carries two first accesses.
    assert result == {"t": pytest.approx(2 * penalty), "u": pytest.approx(penalty)}
    # A bank reached first in an unbounded task is no longer first later.
    assert rows.first_access_ns(facts, {"ddr": ddr}, {"b": "t"}) == {"t": 0.}
