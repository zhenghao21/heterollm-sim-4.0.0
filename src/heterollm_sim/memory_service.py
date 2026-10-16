"""Small shared metrics helper for analytical memory-service envelopes.

The simulator still owns the service-time model.  This helper only derives
observable rates from that model, so a link peak is never reported as the
achieved memory throughput.  All values are analytical unless a caller adds
its own measurement evidence.
"""

from __future__ import annotations

import math
from typing import Optional


# Optional Core observations. An older/analytical controller omitting these
# fields has not measured zero commands or zero occupied command-bus time.
DRAM_CONTROL_COUNT_FIELDS = (
    "activate_commands", "precharge_commands", "read_commands", "write_commands", "refresh_commands",
)
DRAM_CONTROL_TIME_FIELDS = ("row_command_bus_busy_ns", "column_command_bus_busy_ns")
DRAM_OPTIONAL_TIME_FIELDS = ("refresh_wait_ns", "turnaround_wait_ns") + DRAM_CONTROL_TIME_FIELDS
DRAM_OPTIONAL_COUNTER_FIELDS = (
    "read_write_switches", "scheduler_reordered_requests", "scheduler_batches",
    "controller_direction_switches",
) + DRAM_OPTIONAL_TIME_FIELDS + DRAM_CONTROL_COUNT_FIELDS
DRAM_SERVICE_TIME_SCOPE = (
    "service_ns sums per-task/owner completion spans, which may overlap; it is neither elapsed wall time "
    "nor data-bus busy time. resource_totals contains actual per-resource occupied time and bytes."
)


def realtime_memory_metrics(
    logical_bytes: int,
    service_ns: float,
    *,
    physical_bytes: Optional[int] = None,
    bandwidth_ceiling_gb_s: Optional[float] = None,
    queue_wait_ns: float = 0.0,
    request_window_utilization: float = 0.0,
    bottleneck: str = "unknown",
) -> dict:
    """Return derived throughput/utilization fields for one service envelope.

    ``logical_bytes`` is the user-visible payload. ``physical_bytes`` may be
    larger when page reads, RMW, alignment, or write amplification are modeled.
    ``bandwidth_ceiling_gb_s`` is the payload-specific ceiling after direction
    and controller sharing have been resolved; it is deliberately not a link
    bandwidth.  Decimal GB/s equals bytes/ns in the simulator's units.
    """

    if isinstance(logical_bytes, bool) or not isinstance(logical_bytes, int) or logical_bytes < 0:
        raise ValueError("logical_bytes must be a non-negative integer")
    if physical_bytes is None:
        physical_bytes = logical_bytes
    if isinstance(physical_bytes, bool) or not isinstance(physical_bytes, int) or physical_bytes < 0:
        raise ValueError("physical_bytes must be a non-negative integer")
    for name, value in (
        ("service_ns", service_ns),
        ("queue_wait_ns", queue_wait_ns),
        ("request_window_utilization", request_window_utilization),
    ):
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
    if float(request_window_utilization) > 1.0:
        raise ValueError("request_window_utilization must be in [0, 1]")
    if bandwidth_ceiling_gb_s is not None:
        if (isinstance(bandwidth_ceiling_gb_s, bool)
                or not math.isfinite(float(bandwidth_ceiling_gb_s))
                or float(bandwidth_ceiling_gb_s) < 0):
            raise ValueError("bandwidth_ceiling_gb_s must be finite and non-negative")
    service = float(service_ns)
    logical_rate = logical_bytes / service if logical_bytes and service > 0 else 0.0
    physical_rate = physical_bytes / service if physical_bytes and service > 0 else 0.0
    ceiling = float(bandwidth_ceiling_gb_s or 0.0)
    utilization = min(1.0, physical_rate / ceiling) if ceiling > 0 else 0.0
    return {
        "realtime_throughput_gb_s": logical_rate,
        "physical_realtime_throughput_gb_s": physical_rate,
        "bandwidth_ceiling_gb_s": ceiling,
        "bandwidth_utilization": utilization,
        "queue_wait_ns": float(queue_wait_ns),
        "request_window_utilization": float(request_window_utilization),
        "bottleneck": bottleneck,
        "throughput_evidence": "ANALYTICAL_DERIVED",
    }
