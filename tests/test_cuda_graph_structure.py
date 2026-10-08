"""Structural compiler inputs remain independent of observed native timing."""
import json

import pytest

from heterollm_sim.cuda_graph_lifecycle import SOURCE_REVISION
from heterollm_sim.cuda_graph_structure import (
    compare_cuda_graph_structure, load_cuda_graph_structure,
)


def write_program(tmp_path, name, *, dry, key="pointer:a", value="aabb", change=False):
    rows = [{"kind": "node_property", "id": 0, "bytes": value},
            {"kind": "node_property", "id": 1, "bytes": "ffff"}]
    for index in range(3):
        rows.append({"kind": "snapshot", "source_revision": SOURCE_REVISION,
                     "label": f"measured:0:decode:{index}", "device": 0,
                     "context_id": "context:" + key, "graph_key": key,
                     "graph_uid": 0, "n_nodes": 1, "compatible": True,
                     "dry_run": dry, "node_property_refs": [1 if change and index == 2 else 0]})
        # These may appear in a live diagnostic file, but never become part of
        # its structural input representation.
        rows.append({"kind": "decision", "use_cuda_graph": True, "duration_ns": 123,
                     "update_result": "success"})
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_live_decisions_cannot_become_predictor_inputs(tmp_path):
    path = write_program(tmp_path, "live.jsonl", dry=False)
    with pytest.raises(ValueError, match="cannot be prediction inputs"):
        load_cuda_graph_structure(path)
    diagnostic = load_cuda_graph_structure(path, require_dry_run=False)
    with pytest.raises(ValueError, match="cannot be prediction inputs"):
        diagnostic.calls[0].invocation(enabled=True)


def test_dry_input_uses_only_raw_properties_not_decisions(tmp_path):
    program = load_cuda_graph_structure(write_program(tmp_path, "dry.jsonl", dry=True))
    call = program.for_label("measured:0:decode:0")[0]
    invocation = call.invocation(enabled=True)
    assert invocation.node_properties == (b"\xaa\xbb",)
    assert invocation.update_result is None
    assert not hasattr(call, "duration_ns")


def test_structure_verification_compares_equality_not_cross_process_addresses(tmp_path):
    dry = load_cuda_graph_structure(write_program(tmp_path, "dry.jsonl", dry=True))
    live = load_cuda_graph_structure(write_program(tmp_path, "live.jsonl", dry=False,
        key="pointer:b", value="ccdd"), require_dry_run=False)
    assert compare_cuda_graph_structure(dry, live)["qualified"]
    changed = load_cuda_graph_structure(write_program(tmp_path, "changed.jsonl", dry=False,
        key="pointer:c", value="1122", change=True), require_dry_run=False)
    comparison = compare_cuda_graph_structure(dry, changed)
    assert not comparison["qualified"]
    assert comparison["mismatches"][0]["call_index"] == 2


def test_incomplete_property_dictionary_fails_closed(tmp_path):
    path = write_program(tmp_path, "dry.jsonl", dry=True)
    rows = path.read_text(encoding="utf-8").splitlines()
    rows.pop(0)
    path.write_text("\n".join(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="undefined node properties"):
        load_cuda_graph_structure(path)
