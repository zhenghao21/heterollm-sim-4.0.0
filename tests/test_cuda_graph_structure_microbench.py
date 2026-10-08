"""Synthetic graph recipes must exactly match the serving descriptor."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from run_cuda_graph_structure_microbench import typed_chain_nodes, topology_key
from heterollm_sim.cuda_graph_serving import cuda_graph_topology_descriptor


def structure():
    kernel = {"type": 0, "function": "unused-address", "grid": [1, 1, 1],
              "block": [1, 1, 1], "shared_bytes": 0}
    return {"capture_only": True,
        "nodes": [{"id": "a", **kernel}, {"id": "b", **kernel},
                  {"id": "c", "type": 1, "copy": {"src_memory_type": 2, "dst_memory_type": 2,
                      "width_bytes": 8192, "height": 1, "depth": 1, "src_pitch": 0, "dst_pitch": 0}}],
        "edges": [["a", "b"], ["b", "c"]],
        "edge_data": ["0100010000000000", "0000000000000000"]}


def test_synthetic_recipe_matches_predictor_without_kernel_addresses():
    row = structure()
    key = topology_key(typed_chain_nodes(row))
    assert key == cuda_graph_topology_descriptor(row)["topology"]
    assert "unused-address" not in key
    assert "8192" in key
    assert "0100010000000000" in key


@pytest.mark.parametrize("mutation", ["live", "unknown_edge", "missing_copy", "host_copy", "branch"])
def test_synthetic_recipe_rejects_unmeasured_semantics(mutation):
    row = structure()
    if mutation == "live":
        row["capture_only"] = False
    elif mutation == "unknown_edge":
        row["edge_data"][0] = "0200010000000000"
    elif mutation == "missing_copy":
        row["nodes"][2].pop("copy")
    elif mutation == "host_copy":
        row["nodes"][2]["copy"]["src_memory_type"] = 1
    else:
        row["edges"][1] = ["a", "c"]
    with pytest.raises(ValueError):
        typed_chain_nodes(row)
