"""Native destination copies preserve deepcopy ownership and callback order."""
import copy
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.communication import RouteHop
from heterollm_sim.kv_transfer import _copy_plan_destinations, _copy_plan_states


def row():
    return {"component_id": "hbf0", "buffer_id": "kv:r0:L0:k", "generation": 0,
            "offset_bytes": 0, "size_bytes": 4096, "physical_owner": "hbf0.memory"}


def test_authored_keep_alive_callback_can_replace_the_enclosing_tuple_memo():
    destinations = (row(),)
    sentinel = object()
    memo = {}

    class KeepAlive(list):
        def append(self, item):
            super().append(item)
            if item is destinations[0]:
                memo[id(destinations)] = sentinel

    memo[id(memo)] = KeepAlive()
    assert _copy_plan_destinations(destinations, memo) is sentinel
    assert memo[id(destinations)] is sentinel


def test_rows_are_independent_ordered_and_keep_float_bits_and_snapshot_values():
    original = row()
    original.update(negative_zero=-0.0,
                    payload_nan=struct.unpack("!d", bytes.fromhex("7ff8000000000042"))[0])
    rollback = {"public": [1, 2]}
    saved = _copy_plan_states({4: {"destinations": (original,), "rollback_snapshot": rollback}})
    copied = saved[4]["destinations"][0]
    assert copied is not original and list(copied) == list(original)
    for name in ("negative_zero", "payload_nan"):
        assert struct.pack("!d", copied[name]) == struct.pack("!d", original[name])
    assert saved[4]["rollback_snapshot"] is rollback
    original["size_bytes"] = 999
    for _ in range(2):
        restored = _copy_plan_states(saved)
        assert restored[4]["destinations"][0]["size_bytes"] == 4096
        restored[4]["destinations"][0]["size_bytes"] = 777
        assert saved[4]["destinations"][0]["size_bytes"] == 4096


@pytest.mark.parametrize("alias_before", (False, True))
def test_one_memo_preserves_row_and_tuple_aliases_across_fields_and_plans(alias_before):
    original = row()
    destinations = (original, original)
    first = ({"alias": original, "destinations": destinations} if alias_before else
             {"destinations": destinations, "alias": original})
    saved = _copy_plan_states({0: first, 1: {"destinations": destinations, "alias": original}})
    left, right = saved[0], saved[1]
    assert left["destinations"] is right["destinations"]
    assert left["alias"] is right["alias"] is left["destinations"][0] is left["destinations"][1]
    assert left["alias"] is not original


@pytest.mark.parametrize("base", (tuple, list, dict))
def test_container_subclasses_keep_authored_deepcopy(base):
    events = []
    class Authored(base):
        def __deepcopy__(self, memo):
            events.append("authored")
            return ("authored-copy",)
    value = Authored({"size_bytes": 5}) if base is dict else Authored([row()])
    destinations = (value,) if base is dict else value
    copied = _copy_plan_destinations(destinations, {})
    expected = (("authored-copy",),) if base is dict else ("authored-copy",)
    assert copied == expected
    assert events == ["authored"]


def test_unknown_nested_mutables_and_route_hop_use_original_deepcopy():
    hop = RouteHop("pd0", "pd_host", "gpu0", "pd0:wire", 800.0, 123.0)
    mutable = [1, {"bytes": 2}]
    original = row()
    original.update(hop=hop, authored=mutable)
    copied = _copy_plan_destinations((original, original), {})
    assert copied[0] is copied[1] and copied[0] is not original
    assert copied[0]["hop"] == hop and copied[0]["hop"] is not hop
    assert copied[0]["authored"] == mutable and copied[0]["authored"] is not mutable
    assert copied[0]["authored"][1] is not mutable[1]


def test_partial_native_prefix_then_unknown_cycle_preserves_tuple_recursion():
    first = row()
    back = []
    value = (first, back)
    back.append(value)
    copied = _copy_plan_destinations(value, {})
    assert copied is not value and copied[0] is not first
    assert copied[1] is not back and copied[1][0] is copied


def test_earlier_custom_field_can_seed_scalar_memo_and_mutate_later_row():
    original = row()
    value = original["buffer_id"]
    class SeedMemo:
        def __deepcopy__(self, memo):
            original["size_bytes"] = 777
            memo[id(value)] = "memo-substitute"
            return "hook-copy"
    copied = _copy_plan_states({0: {"hook": SeedMemo(), "destinations": (original,)}})
    assert copied[0]["hook"] == "hook-copy"
    assert copied[0]["destinations"][0]["buffer_id"] == "memo-substitute"
    assert copied[0]["destinations"][0]["size_bytes"] == 777


def test_unknown_callback_after_native_prefix_keeps_call_order_and_failure():
    first = row()
    calls = []
    class Rejected:
        def __deepcopy__(self, memo):
            calls.append("rejected")
            assert memo[id(first)] is not first
            assert memo[id(first)]["size_bytes"] == 4096
            raise RuntimeError("authored rejection")
    with pytest.raises(RuntimeError, match="authored rejection"):
        _copy_plan_destinations((first, {"custom": Rejected()}), {})
    assert calls == ["rejected"] and first["size_bytes"] == 4096


def test_replaced_copy_dispatch_is_observed(monkeypatch):
    monkeypatch.setitem(copy._deepcopy_dispatch, int, lambda value, memo: value + 3)
    copied = _copy_plan_destinations((row(),), {})
    assert copied[0]["size_bytes"] == 4099


def test_replaced_deepcopy_is_called_once_for_the_destination_field(monkeypatch):
    calls = []
    def authored(value, memo):
        calls.append(value)
        return "authored-field"
    monkeypatch.setattr(copy, "deepcopy", authored)
    destinations = (row(),)
    assert _copy_plan_destinations(destinations, {}) == "authored-field"
    assert calls == [destinations]
