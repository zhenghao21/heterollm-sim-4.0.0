import importlib.util
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("typed_activity", TOOLS / "run_cuda_graph_typed_activity_microbench.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def evidence():
    nodes = [[0, 0, 0], [1, 0, 8192], [0, 0, 0], [0, 1, 0]]
    samples = []
    for mode in ("ordinary", "first_launch", "replay"):
        for _ in range(3):
            samples.append({"mode": mode, "device_event_ns": 1000,
                            "activities": [[0, 0, 100, 0, 1 if mode != "ordinary" else 0],
                                           [1, 150, 500, 8192, 2 if mode != "ordinary" else 0],
                                           [0, 550, 800, 0, 3 if mode != "ordinary" else 0],
                                           [0, 750, 900, 0, 4 if mode != "ordinary" else 0]]})
    return {"schema": "heterollm.cuda-typed-activity/v1", "target_llm_latency_used": False,
            "activity_enabled": True, "node_count": 4, "nodes": nodes, "samples": samples}


def test_copy_execution_is_separate_from_boundary_gap():
    summary = module.summarize_activity(evidence())
    edges = summary["edge_measurements"]
    assert all(item["mean_signed_gap"]["median_ns"] == 50 for item in edges if item["to_kind"] == 1 or item["from_kind"] == 1)
    assert all(item["to_copy_bytes"] == 8192 for item in edges if item["to_kind"] == 1)


def test_pdl_whole_kernel_overlap_disallows_serial_cost():
    summary = module.summarize_activity(evidence())
    pdl = [item for item in summary["edge_measurements"] if item["incoming_dependency"] == 1]
    assert all(item["mean_signed_gap"]["median_ns"] == -50 for item in pdl)
    assert all(not item["serial_additive_candidate"] for item in pdl)


@pytest.mark.parametrize("mutation", ["missing", "wrong_copy", "wrong_graph"])
def test_incomplete_or_wrong_activity_is_rejected(mutation):
    raw = evidence()
    if mutation == "missing": raw["samples"][0]["activities"].pop()
    if mutation == "wrong_copy": raw["samples"][0]["activities"][1][3] = 4096
    if mutation == "wrong_graph": raw["samples"][0]["activities"][0][4] = 10
    with pytest.raises(ValueError):
        module.summarize_activity(raw)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), True])
def test_bad_profiler_control_boundary_is_rejected(duration):
    raw = evidence()
    raw["samples"][0]["device_event_ns"] = duration
    with pytest.raises(ValueError):
        module.summarize_activity(raw)


def test_duplicated_graph_node_is_not_mistaken_for_complete_inventory():
    raw = evidence()
    raw["samples"][3]["activities"][1][4] = 1
    with pytest.raises(ValueError, match="duplicate"):
        module.summarize_activity(raw)
