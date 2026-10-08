"""Dispatch ownership must be a complete source-observed graph partition."""
import json

import pytest

from heterollm_sim.cuda_graph_dispatch import SCHEMA, bind_source_dispatches, load_source_dispatches
from heterollm_sim.cuda_graph_lifecycle import SOURCE_REVISION
from heterollm_sim.cuda_graph_serving import cuda_graph_topology_descriptor


def tensor(name="attn_norm-0"):
    return {"name": name, "op": "MUL", "type": "f32", "tensor": "1", "data": "2",
            "ne": [128, 1, 1, 1], "nb": [4, 512, 512, 512]}


def graph():
    return {"kind": "cuda_structure", "capture_only": True, "source_revision": SOURCE_REVISION,
        "call_id": 1, "nodes": [{"id": key, "type": 0, "function": "fn", "grid": [1, 1, 1],
        "block": [32, 1, 1], "shared_bytes": 0, "cooperative": 0} for key in "abc"],
        "edges": [["a", "b"], ["b", "c"]], "edge_data": ["0000000000000000", "0100010000000000"]}


def dispatch(first, last, before, after):
    return {"kind": "source_dispatch", "schema": SCHEMA, "source_revision": SOURCE_REVISION,
        "capture_only": True, "call_id": 1, "first_ggml_index": first, "last_ggml_index": last,
        "before": before, "after": after, "source_nodes": [{"ggml_index": i,
            "output": tensor(f"node-{i}"), "sources": [{"slot": 0, "tensor": tensor("blk.0.attn_norm.weight")}]}
            for i in range(first, last + 1)]}


def rows():
    # Two fused GGML operators emit one CUDA kernel; another emits no kernel;
    # the final single GGML operator emits two actual CUDA kernels.
    return [dispatch(1, 2, [], ["a"]), dispatch(4, 4, ["a"], ["a"]),
            dispatch(7, 7, ["a"], ["c"])]


def test_exact_many_to_many_ownership_without_proportional_mapping():
    result = bind_source_dispatches(cuda_graph_topology_descriptor(graph()), rows(), ggml_node_count=8)
    assert [group["node_ids"] for group in result] == [("a",), (), ("b", "c")]
    assert [group["node_count"] for group in result] == [1, 0, 2]
    assert [group["first_node_ordinal"] for group in result] == [0, 1, 1]
    assert len(result[0]["source_nodes"]) == 2
    assert result[-1]["nodes"][-1]["id"] == "c"


@pytest.mark.parametrize("mutation", ["missing", "overlap", "ggml_gap", "no_end", "ambiguous",
    "uncovered", "source_missing", "source_shape", "source_slots", "live", "revision", "mixed_call"])
def test_rejects_unproven_ownership(mutation):
    value = rows()
    if mutation == "missing": value = []
    elif mutation == "overlap": value[1]["first_ggml_index"] = 2
    elif mutation == "ggml_gap": value[0]["source_nodes"].pop()
    elif mutation == "no_end": value[-1]["after"] = ["unknown"]
    elif mutation == "ambiguous": value[-1]["before"] = ["a", "b"]
    elif mutation == "uncovered": value[-1]["after"] = ["b"]
    elif mutation == "source_missing": value[0]["source_nodes"][0].pop("output")
    elif mutation == "source_shape": value[0]["source_nodes"][0]["output"]["ne"] = [1]
    elif mutation == "source_slots": value[0]["source_nodes"][0]["sources"] *= 2
    elif mutation == "live": value[0]["capture_only"] = False
    elif mutation == "revision": value[0]["source_revision"] = "different"
    else: value[-1]["call_id"] = 2
    with pytest.raises(ValueError):
        bind_source_dispatches(cuda_graph_topology_descriptor(graph()), value, ggml_node_count=8)


def test_capture_reader_retains_source_ownership_and_rejects_legacy_missing_groups(tmp_path):
    snapshot = {"kind": "snapshot", "source_revision": SOURCE_REVISION, "call_id": 1,
        "label": "measured:0:decode:1", "n_nodes": 8, "capture_only": True, "dry_run": False}
    path = tmp_path / "capture.jsonl"
    def write(records):
        path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    write([snapshot, *rows(), graph()])
    groups = load_source_dispatches(path)[snapshot["label"]]
    assert sum(group["node_count"] for group in groups) == 3
    write([snapshot, graph()])
    with pytest.raises(ValueError, match="lacks source dispatch"):
        load_source_dispatches(path)
    write([snapshot, *rows()])
    with pytest.raises(ValueError, match="incomplete"):
        load_source_dispatches(path)
