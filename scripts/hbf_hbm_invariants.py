"""Small, hand-checkable invariants for the HBM/HBF physical backends.

These checks intentionally exercise the cores directly.  They are independent
of the serving acceptance matrix so a timing regression is visible before it
is hidden by request scheduling or placement policy.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import (
    AccessRequest,
    Operation,
)
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.nand_core import NandCore
from heterollm_sim.physical_presets import resolve_physical_config


def _ideal_hbm():
    return resolve_physical_config("research_hbm_ideal")


def _ideal_hbf():
    # The NAND path keeps its page/granularity accounting, but with a page and
    # host granularity equal to one DRAM burst and zero media/front overhead.
    # Internal bandwidth is made much larger than the host link, so the
    # externally visible read service is the same 64 GB/s link.
    return resolve_physical_config("research_hbf_ideal")


def _ratio(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def check_ideal_read_equivalence() -> dict[str, Any]:
    """Same read link/granularity and no media overhead give equal timing."""

    request = AccessRequest("ideal-read", Operation.READ, 0, 4096)
    hbm_result = DramCore(_ideal_hbm(), ResourceTimeline()).submit(request)
    hbf_result = NandCore(_ideal_hbf(), ResourceTimeline()).submit(request)
    timing_error = _ratio(hbm_result.completion_ns, hbf_result.completion_ns)
    passed = (
        hbm_result.logical_bytes == hbf_result.logical_bytes == request.byte_count
        and hbm_result.transfer_bytes == hbf_result.host_transfer_bytes == request.byte_count
        and timing_error <= 0.01
    )
    return {
        "id": "ideal_read_equivalence",
        "passed": passed,
        "metrics": {
            "logical_bytes": hbm_result.logical_bytes,
            "hbm_completion_ns": hbm_result.completion_ns,
            "hbf_completion_ns": hbf_result.completion_ns,
            "timing_relative_error": timing_error,
            "hbm_transfer_bytes": hbm_result.transfer_bytes,
            "hbf_host_transfer_bytes": hbf_result.host_transfer_bytes,
            "tolerance": 0.01,
        },
        "reason": (
            "Both paths use 64-byte granularity, 64 GB/s read service and zero "
            "latency/media overhead; the HBF internal leg is intentionally not "
            "the bottleneck."
        ),
    }


def check_aggregate_execution_consistency() -> dict[str, Any]:
    """One contiguous request and its ordered burst requests agree."""

    config = _ideal_hbm()
    total_bytes = 128 * config.burst_bytes
    aggregate = DramCore(config, ResourceTimeline(), capture_details=False).submit(
        AccessRequest("aggregate", Operation.READ, 0, total_bytes)
    )
    split_core = DramCore(config, ResourceTimeline(), capture_details=False)
    split = tuple(
        split_core.submit(AccessRequest(f"burst-{index}", Operation.READ, index * config.burst_bytes, config.burst_bytes))
        for index in range(128)
    )
    split_last = max(item.completion_ns for item in split)
    split_bytes = sum(item.logical_bytes for item in split)
    timing_error = _ratio(aggregate.completion_ns, split_last)
    passed = (
        aggregate.logical_bytes == split_bytes == total_bytes
        and aggregate.transfer_bytes == sum(item.transfer_bytes for item in split)
        and timing_error <= 0.01
    )
    return {
        "id": "aggregate_execution_consistency",
        "passed": passed,
        "metrics": {
            "request_count": len(split),
            "aggregate_logical_bytes": aggregate.logical_bytes,
            "split_logical_bytes": split_bytes,
            "aggregate_transfer_bytes": aggregate.transfer_bytes,
            "split_transfer_bytes": sum(item.transfer_bytes for item in split),
            "aggregate_completion_ns": aggregate.completion_ns,
            "split_completion_ns": split_last,
            "timing_relative_error": timing_error,
            "tolerance": 0.01,
        },
    }


def check_long_context_event_bound() -> dict[str, Any]:
    """Large requests retain scalar timing without materializing every event."""

    dram_config = _ideal_hbm()
    dram_config = type(dram_config)(
        **{**dram_config.__dict__, "max_expanded_segments": 16}
    )
    dram_bytes = 4 * 1024 * 1024
    dram = DramCore(dram_config, ResourceTimeline(), capture_details=False).submit(
        AccessRequest("long-hbm", Operation.READ, 0, dram_bytes)
    )
    hbf_config = _ideal_hbf()
    hbf_config = type(hbf_config)(
        **{**hbf_config.__dict__, "max_expanded_segments": 16}
    )
    hbf = NandCore(hbf_config, ResourceTimeline(), capture_details=False).submit(
        AccessRequest("long-hbf", Operation.READ, 0, 4 * 1024 * 1024)
    )
    passed = (
        dram.logical_bytes == dram_bytes
        and hbf.logical_bytes == dram_bytes
        and dram.counters.get("details_truncated") is True
        and hbf.counters.get("details_truncated") is True
        and not dram.stages
        and not hbf.stages
        and dram.completion_ns > 0.0
        and hbf.completion_ns > 0.0
    )
    return {
        "id": "long_context_event_bound",
        "passed": passed,
        "metrics": {
            "bytes": dram_bytes,
            "dram_burst_count": dram.counters.get("burst_count"),
            "nand_page_count": hbf.counters.get("page_count"),
            "dram_stage_count": len(dram.stages),
            "nand_stage_count": len(hbf.stages),
            "dram_details_truncated": dram.counters.get("details_truncated"),
            "nand_details_truncated": hbf.counters.get("details_truncated"),
            "dram_completion_ns": dram.completion_ns,
            "nand_completion_ns": hbf.completion_ns,
        },
    }


def run_invariants() -> dict[str, Any]:
    checks = [
        check_ideal_read_equivalence(),
        check_aggregate_execution_consistency(),
        check_long_context_event_bound(),
    ]
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


if __name__ == "__main__":
    import json

    print(json.dumps(run_invariants(), indent=2, allow_nan=False))
