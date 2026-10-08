import importlib.util
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("queue_probe", TOOLS / "run_cuda_dispatch_queue_microbench.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
analysis_spec = importlib.util.spec_from_file_location("queue_analysis", TOOLS / "analyze_cuda_dispatch_queue_microbench.py")
analysis = importlib.util.module_from_spec(analysis_spec)
analysis_spec.loader.exec_module(analysis)


def evidence():
    case = {"node_count": 3, "blocks": 1, "threads": 32, "extra_parameter_bytes": 0,
            "observe_host": True, "requested_bodies_ns": [125, 750, 1250]}
    raw = {"schema": "heterollm.cuda-dispatch-queue/v1", "target_llm_latency_used": False,
           **{key: value for key, value in case.items() if key != "observe_host"},
           "host_observation": True, "cuda_parameter_extent_bytes": 32,
           "samples": [{"device_event_ns": 7000, "node_begin_ns": [0, 2048, 4096], "node_end_ns": [200, 3000, 5500],
                        "host_enqueue_begin_ns": [0, 900, 1800], "host_enqueue_end_ns": [800, 1700, 2600],
                        "host_release_ns": 3000} for _ in range(3)]}
    return raw, case


def test_cpu_and_gpu_times_are_separate_domains():
    raw, case = evidence()
    result = module.summarize(raw, case)
    assert result["device_span"]["median_ns"] == 5500
    assert result["host_submission_span"]["median_ns"] == 2600
    assert result["prediction_qualified"] is False


@pytest.mark.parametrize("mutation", ["missing", "inverted", "release_early", "wrong_parameter_bytes", "wrong_body"])
def test_incomplete_or_mismatched_queue_probe_is_rejected(mutation):
    raw, case = evidence()
    if mutation == "missing": raw["samples"][0]["host_enqueue_end_ns"].pop()
    if mutation == "inverted": raw["samples"][0]["host_enqueue_begin_ns"][1] = 700
    if mutation == "release_early": raw["samples"][0]["host_release_ns"] = 2500
    if mutation == "wrong_parameter_bytes": raw["cuda_parameter_extent_bytes"] = 48
    if mutation == "wrong_body": raw["requested_bodies_ns"] = [125, 500, 1250]
    with pytest.raises(ValueError):
        module.summarize(raw, case)


def test_training_and_holdout_are_declared_without_observed_latency():
    plan = module.create_plan()
    assert plan["target_llm_latency_used"] is False
    assert len(plan["execution_order"]) == len(plan["cases"])
    assert set(plan["execution_order"]) == set(range(len(plan["cases"])))
    training = {case["requested_bodies_ns"][0] for case in plan["cases"] if case["split"] == "train_threshold"}
    held = {case["requested_bodies_ns"][0] for case in plan["cases"] if case["split"] == "holdout_threshold"}
    assert training.isdisjoint(held)


def training_entry(body):
    case = {"case_id": f"train_{body}", "split": "train_threshold", "extra_parameter_bytes": 0,
            "requested_bodies_ns": [body] * 128, "node_count": 128}
    starts, ends = [0], [body + 192]
    for i in range(1, 128):
        pulse = 0 if i % 11 else (20000 if (i // 11) % 2 else 5000)
        start = ends[-1] + analysis.regular_gap(body + 192, 1000) + pulse
        starts.append(start); ends.append(start + body + 192)
    raw = {"samples": [{"node_begin_ns": starts, "node_end_ns": ends} for _ in range(3)]}
    return case, raw


def test_holdout_cannot_be_passed_as_training():
    case, raw = training_entry(0)
    case["split"] = "holdout_threshold"
    with pytest.raises(ValueError, match="training cases only"):
        analysis.fit_training([(case, raw)])


def test_training_recovers_period_without_any_holdout():
    fitted = analysis.fit_training([training_entry(0), training_entry(1000)])
    assert fitted["tail_lower_exclusive_ns"] == 856
    assert fitted["tail_upper_inclusive_ns"] == 1856
    assert fitted["packet_models"][0]["packet_period_nodes"] == 11


def test_unmeasured_argument_footprint_is_not_nearest_matched():
    fitted = analysis.fit_training([training_entry(0), training_entry(1000)])
    case, raw = training_entry(1000)
    case["extra_parameter_bytes"] = 1024
    result = analysis.predict_case(raw, case, fitted)
    assert result["status"] == "unsupported_parameter_extent"
