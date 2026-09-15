from types import SimpleNamespace
from heterollm_sim.planner import summarize_host_gemm_offload


def task(phase, applied=None, proven=False):
    meta = {"phase": phase}
    if applied is not None:
        meta.update(host_gemm_offload_applied=applied,
                    host_gemm_offload_reason="offloaded" if applied else "below_threshold",
                    host_gemm_offload_provenance={"native_dispatch_proven": proven})
    return SimpleNamespace(metadata=meta)


def test_counts_only_physical_gemm_once_and_keeps_unknown_evidence():
    result = summarize_host_gemm_offload([
        task("kernel_launch", True), task("h2d", True), task("memory", True),
        task("gpu_gemm", True), task("gpu_gemm", True, True),
        task("cpu_gemm", False), task("gpu_gemm"),
    ])
    assert result["gemm_tasks"] == 4
    assert result["audited_tasks"] == 3
    assert result["applied_tasks"] == 2
    assert result["conditional_tasks"] == 1
    assert result["native_dispatch_proven_tasks"] == 1
    assert result["uncovered_tasks"] == 1
    assert result["reason_counts"] == {"below_threshold": 1, "offloaded": 2}


def test_empty_does_not_manufacture_any_executed_gemm():
    result = summarize_host_gemm_offload([])
    assert result["gemm_tasks"] == result["applied_tasks"] == 0
    assert result["reason_counts"] == {}
