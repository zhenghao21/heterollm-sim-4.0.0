"""Cached row-hit estimates must stay exact across maintenance and contention."""
import math

import pytest

from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import assert_same, configuration


@pytest.mark.parametrize("clock", [.625, .7])
@pytest.mark.parametrize("mapping", ["bank_interleaved", "row_local"])
@pytest.mark.parametrize("refresh", [False, True])
def test_row_hit_estimate_rechecks_maintenance_and_preserves_following_state(clock, mapping, refresh):
    config = configuration(command_clock_ns=clock, data_rate_mt_s=4000. / clock,
        read_latency_ns=8. * clock, write_latency_ns=4. * clock,
        address_mapping=mapping, maximum_active_ns=180., four_activate_window_ns=20.,
        refresh_policy="all_bank" if refresh else "disabled",
        refresh_interval_ns=400. if refresh else 0., refresh_cycle_ns=40. if refresh else 0.)
    reference = DramCore(config, capture_details=False)
    numeric = DramCore(config, capture_details=False)
    span = config.lane_count * config.bank_count * config.row_bytes
    # Repeated requests hit the existing rows, alternate DQ direction, then
    # cross active-row and refresh deadlines before hitting the new rows.
    requests = (
        ("read", 0, 4096, 0.),
        ("read", 0, 128, 0.),
        ("write", 0, 128, 0.),
        ("read", span, 4096, 15.),
        ("write", span, 256, math.nextafter(400., math.inf)),
        ("read", span, 256, 400.),
        ("read", 0, 512, 1200.),
        ("write", 0, 256, 1.),
    )
    for index, (operation, address, count, arrival) in enumerate(requests):
        if index == 4:
            for core in (reference, numeric):
                # A shared CA port and a PC data pipe can independently delay
                # the estimate beyond the request's external arrival time.
                core.timeline.reserve("dram:column_command:0:0", 0., 450.)
                core.timeline.reserve("dram:data:0", 0., 460., direction="read")
        assert_same(reference, numeric,
                    AccessRequest(str(index), operation, address, count, arrival))
