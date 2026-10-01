import hashlib
import json

import pytest

from heterollm_sim.runtime_residual import (
    RuntimeResidualCalibration,
    load_runtime_residual_calibration,
)


def test_runtime_residual_separates_ordinary_graph_and_kernel_count():
    calibration = RuntimeResidualCalibration(
        "gpu", "runtime", "arch", 10.0, 7.0, "synthetic", qualified=True
    )
    ordinary = calibration.cost(4, captured=False)
    graph = calibration.cost(4, captured=True)
    assert ordinary["service_ns"] == 40.0
    assert ordinary["submission_count"] == 4
    assert graph["service_ns"] == 7.0
    assert graph["submission_count"] == 1


def test_graph_launch_phase_uses_runtime_submission_when_device_enqueue_missing():
    from heterollm_sim.kernel_model import graph_launch_phases
    from heterollm_sim.kernel_model import KernelModelProfile
    model = KernelModelProfile(
        "gpu", "runtime", "arch", launch_ns=None, graph_launch_ns=7.0,
        runtime_submission_ns=3.0, launch_evidence="synthetic",
        runtime_submission_evidence="runtime_probe", graph_enabled=True,
    )
    phase = graph_launch_phases(model, "frontend", 4, captured=False)
    assert phase.demands[0].service_ns == 12.0
    replay = graph_launch_phases(model, "frontend", 4, captured=True)
    assert replay.demands[0].service_ns == 7.0


def test_runtime_residual_import_rejects_llm_target(tmp_path):
    path = tmp_path / "residual.json"
    raw = {
        "schema": "heterollm.runtime-residual/v1",
        "source_kind": "independent_synthetic_runtime_microbenchmark",
        "target_llm_latency_used": False,
        "measurement_boundary": "host_submission_cuda_event_pair",
        "hardware_id": "gpu",
        "runtime_id": "runtime",
        "architecture": "arch",
        "samples": [{
            "kernel_count": 1,
            "ordinary_durations_ns": [10, 11, 10],
            "graph_durations_ns": [7, 8, 7],
        }],
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_runtime_residual_calibration(path).ordinary_launch_ns == pytest.approx(10)
    raw["qualified"] = True
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_runtime_residual_calibration(path).qualified is False
    raw["target_llm_latency_used"] = True
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="target LLM"):
        load_runtime_residual_calibration(path)


def test_runtime_residual_import_keeps_count_dependent_evidence_diagnostic(tmp_path):
    path = tmp_path / "residual.json"
    raw = {
        "schema": "heterollm.runtime-residual/v1",
        "source_kind": "independent_synthetic_runtime_microbenchmark",
        "target_llm_latency_used": False,
        "measurement_boundary": "host_submission_cuda_event_pair",
        "hardware_id": "gpu", "runtime_id": "runtime", "architecture": "arch",
        "samples": [
            {"kernel_count": 1, "ordinary_durations_ns": [10, 10], "graph_durations_ns": [7, 7]},
            {"kernel_count": 2, "ordinary_durations_ns": [30, 30], "graph_durations_ns": [7, 7]},
        ],
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    calibration = load_runtime_residual_calibration(path)
    assert calibration.qualified is False


def test_attention_level2_is_phase_specific_and_fail_closed():
    from heterollm_sim.attention_level2 import attention_level2_status
    from heterollm_sim.kernel_model import llama_blackwell_analytical_profile

    prefill = attention_level2_status("prefill", mask="causal")
    decode = attention_level2_status("decode", mask="none")
    assert prefill["phase"] == "prefill" and not prefill["accepted"]
    assert decode["phase"] == "decode" and not decode["accepted"]
    assert attention_level2_status("decode", mask="none", stream_k=True)["accepted"] is False
    profile = llama_blackwell_analytical_profile("gpu", "runtime", calibrated=True)
    assert {k.kernel_family for k in profile.kernels if k.operator == "attention"} == {
        "flash_attention_prefill_l2", "paged_attention_decode_l2"
    }
