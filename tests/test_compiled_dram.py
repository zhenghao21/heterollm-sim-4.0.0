from copy import deepcopy
from dataclasses import replace

import pytest

from heterollm_sim._compiled_dram import execute_compiled
from heterollm_sim.dram_core import DramCore, _BankState
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest, DramConfig, Operation


def _config(**changes):
    return replace(DramConfig(data_lanes=2, banks_per_group=2, rows_per_bank=32),
                   capacity_bytes=None, **changes)


@pytest.mark.parametrize("case", ["multiple_slots", "shared_list", "nonfinite", "negative_row",
                                  "byte_overflow", "geometry_overflow", "inexact_burst", "custom_calendar"])
def test_unsupported_compiled_state_falls_back_without_mutation(case):
    config = _config()
    if case == "geometry_overflow":
        config = _config(rows_per_bank=1 << 63)
    elif case == "inexact_burst":
        config = _config(burst_bytes=(1 << 53) + 8, row_bytes=((1 << 53) + 8) * 2,
                         data_lanes=1, banks_per_group=1, rows_per_bank=1)
    core = DramCore(config, capture_details=False)
    if case == "multiple_slots":
        core.timeline.lane_available["dram:data:0"] = [0.0, 1.0]
    elif case == "shared_list":
        slots = [0.0]
        core.timeline.lane_available.update({"dram:data:0": slots, "dram:data:1": slots})
    elif case == "nonfinite":
        core.timeline.ready_ns["dram:data:0"] = float("inf")
    elif case == "negative_row":
        core._banks["dram:bank:0:0:0:0:0:0"] = _BankState(-1, 0.0)
    elif case == "byte_overflow":
        core.timeline.bytes_moved["dram:data:0"] = (1 << 63) - 1
    elif case == "custom_calendar":
        class CustomCalendar(ResourceTimeline):
            pass
        core.timeline = CustomCalendar()
    request = AccessRequest("fallback", Operation.READ, 0, config.burst_bytes * 2, 0.0)
    before_banks, before_timeline = deepcopy(core._banks), deepcopy(core.timeline)
    assert execute_compiled(core, request) is None
    assert core._banks == before_banks
    assert core.timeline == before_timeline


def test_compiled_commit_preserves_existing_calendar_identity_and_lazy_bank_keys():
    core = DramCore(_config(), capture_details=False)
    slots = [100.0]
    core.timeline.lane_available["dram:data:0"] = slots
    core.timeline.lane_available["gpu:compute"] = [10.0, 20.0]
    core.timeline.ready_ns["gpu:compute"] = 10.0
    request = AccessRequest("one-bank", Operation.READ, 0, 64, 0.0)
    result = execute_compiled(core, request)
    assert result.counters["compiled_float64"] is True
    assert core.timeline.lane_available["dram:data:0"] is slots
    assert core.timeline.lane_available["gpu:compute"] == [10.0, 20.0]
    assert set(core._banks) == {"dram:bank:0:0:0:0:0:0"}
    assert core.timeline._touched == {"dram:bank:0:0:0:0:0:0", "dram:command:0", "dram:data:0"}
    assert "dram:data:1" not in core.timeline.ready_ns
    assert "dram:data:1" not in core.timeline.bytes_moved
