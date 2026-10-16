"""Clock-grid arithmetic without rounding external timestamps backwards."""
from __future__ import annotations

import math


def align_clock(value, quantum):
    """First represented edge at or after an arbitrary timestamp, strictly."""
    tick = math.floor(value / quantum)
    edge = tick * quantum
    if edge < value:
        edge = (tick + 1) * quantum
    return edge


def clock_after(value, delay, quantum):
    """Add internal whole ticks as integers; preserve non-grid inputs.

    Exact membership, rather than a tolerance, is intentional: a caller's
    nextafter(edge, +inf) is not an edge. Only two already represented grid
    values can use integer tick addition. Analog delays retain ns arithmetic.
    """
    if math.isfinite(value) and math.isfinite(delay):
        value_tick, delay_tick = round(value / quantum), round(delay / quantum)
        if value == value_tick * quantum and delay == delay_tick * quantum:
            return (value_tick + delay_tick) * quantum
    return value + delay
