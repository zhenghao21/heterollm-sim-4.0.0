"""The independent probe must separate body execution from observable gaps."""
import importlib.util
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("launch_gap", TOOLS / "run_cuda_graph_launch_gap_microbench.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_execution_body_is_not_launch_cost():
    result = module.sample_intervals({"node_begin_ns": [0, 1100, 6200], "node_end_ns": [1000, 6100, 7200]}, 3)
    assert result["body_sum_ns"] == 7000
    assert result["uncovered_gap_ns"] == 200
    assert result["mean_signed_edge_gap_ns"] == 100


def test_pdl_overlap_is_preserved_and_union_not_sum_is_subtracted():
    result = module.sample_intervals({"node_begin_ns": [0, 80, 200], "node_end_ns": [100, 160, 250]}, 3)
    assert result["signed_edge_gaps_ns"] == [-20, 40]
    assert result["body_sum_ns"] == 230
    assert result["body_union_ns"] == 210
    assert result["uncovered_gap_ns"] == 40
    assert result["overlapping_edge_count"] == 1


@pytest.mark.parametrize("sample", [
    {"node_begin_ns": [0, 2], "node_end_ns": [1]},
    {"node_begin_ns": [0, 2], "node_end_ns": [1, 1]},
    {"node_begin_ns": [0, float("nan")], "node_end_ns": [1, 3]},
    {"node_begin_ns": [0, True], "node_end_ns": [1, 3]},
    {"node_begin_ns": [1, 2], "node_end_ns": [1, 3]},
])
def test_invalid_probe_is_rejected(sample):
    with pytest.raises(ValueError):
        module.sample_intervals(sample, 2)


def experiment():
    return {"schema": module.SCHEMA, "target_llm_latency_used": False, "node_count": 2,
            "samples": [{"mode": mode, "gated": gated, "host_submit_ns": 100,
                         "node_begin_ns": [0, 110], "node_end_ns": [100, 210]}
                        for mode in module.MODES
                        for gated in (False, True) for _ in range(3)]}


def test_gated_and_ungated_boundaries_remain_separate():
    summaries = module.summarize(experiment())
    assert len(summaries) == 8
    assert all(item["mean_signed_edge_gap"]["median_ns"] == 10 for item in summaries)
    assert all(item["serial_additive_candidate"] for item in summaries)


def test_missing_boundary_does_not_fall_back():
    raw = experiment()
    raw["samples"] = [sample for sample in raw["samples"] if sample["mode"] != "replay"]
    with pytest.raises(ValueError, match="unbalanced"):
        module.summarize(raw)


def test_target_model_timing_is_not_accepted():
    raw = experiment()
    raw["target_llm_latency_used"] = True
    with pytest.raises(ValueError, match="not independent"):
        module.summarize(raw)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), True])
def test_invalid_statistics_are_rejected(value):
    with pytest.raises(ValueError, match="non-finite"):
        module.stats([1, value, 3])
