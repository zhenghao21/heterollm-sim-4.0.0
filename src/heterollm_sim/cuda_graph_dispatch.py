"""Exact timing-free ownership of CUDA nodes by source GGML dispatch groups.

The native capture exporter brackets each fused/ordinary source dispatch with
stream-capture frontiers. These frontiers partition the supported typed chain;
neither proportional allocation nor a kernel-count guess is used.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping

from .cuda_graph_lifecycle import SOURCE_REVISION

SCHEMA = "heterollm.cuda-source-dispatch/v1"


def _tensor(value):
    if not isinstance(value, Mapping):
        raise ValueError("source dispatch requires explicit tensor descriptors")
    if not isinstance(value.get("name"), str):
        raise ValueError("source tensor name must be explicit")
    if any(not isinstance(value.get(key), str) or not value[key]
           for key in ("op", "type", "tensor", "data")):
        raise ValueError("source tensor operation/type/identity is incomplete")
    for key in ("ne", "nb"):
        dimensions = value.get(key)
        if (not isinstance(dimensions, list) or len(dimensions) != 4
                or any(type(item) is not int or item < (1 if key == "ne" else 0)
                       for item in dimensions)):
            raise ValueError("source tensor shape/strides are incomplete")


def bind_source_dispatches(descriptor, rows, *, ggml_node_count):
    """Resolve frontier segments to each actual CUDA node, exactly once.

    ``descriptor`` comes from cuda_graph_topology_descriptor, which validates
    the connected typed chain and orders its nodes. No topology is inferred
    from a model's name or its number of simulated operators.
    """
    if type(ggml_node_count) is not int or ggml_node_count < 1:
        raise ValueError("source dispatch requires the exact GGML node count")
    nodes = descriptor.get("ordered_nodes")
    if not nodes or descriptor.get("node_count") != len(nodes):
        raise ValueError("source dispatch requires a validated ordered CUDA chain")
    ids = tuple(node["id"] for node in nodes)
    if len(set(ids)) != len(ids):
        raise ValueError("source dispatch CUDA node identities are ambiguous")
    ordinal = {key: index for index, key in enumerate(ids)}
    if not isinstance(rows, (list, tuple)) or not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("capture structure lacks source dispatch ownership")
    result, cursor, previous_ggml = [], 0, -1
    call_id = rows[0].get("call_id")
    if type(call_id) is not int or call_id < 0:
        raise ValueError("source dispatch call identity is invalid")
    for index, row in enumerate(rows):
        if (row.get("kind") != "source_dispatch" or row.get("schema") != SCHEMA
                or row.get("source_revision") != SOURCE_REVISION
                or row.get("capture_only") is not True or type(row.get("call_id")) is not int
                or row.get("call_id") != call_id):
            raise ValueError("source dispatch lacks matching capture-only provenance")
        first, last = row.get("first_ggml_index"), row.get("last_ggml_index")
        if (type(first) is not int or type(last) is not int
                or not previous_ggml < first <= last < ggml_node_count):
            raise ValueError("source dispatch GGML ranges overlap or exceed the graph")
        before, after = row.get("before"), row.get("after")
        expected_before = [ids[cursor - 1]] if cursor else []
        if before != expected_before:
            raise ValueError("source dispatch frontier is missing, ambiguous or noncontiguous")
        if after == before:
            end = cursor  # Source dispatch can legitimately emit no CUDA node.
        elif (isinstance(after, list) and len(after) == 1
                and after[0] in ordinal and ordinal[after[0]] >= cursor):
            end = ordinal[after[0]] + 1
        else:
            raise ValueError("source dispatch end frontier is missing or ambiguous")
        source_nodes = row.get("source_nodes")
        if (not isinstance(source_nodes, list) or len(source_nodes) != last - first + 1
                or any(not isinstance(node, Mapping) for node in source_nodes)
                or [node.get("ggml_index") for node in source_nodes] != list(range(first, last + 1))):
            raise ValueError("source dispatch must describe every fused GGML node")
        for source in source_nodes:
            _tensor(source.get("output"))
            inputs = source.get("sources")
            if not isinstance(inputs, list):
                raise ValueError("source dispatch tensor inputs are missing")
            slots = []
            for item in inputs:
                if (not isinstance(item, Mapping) or type(item.get("slot")) is not int
                        or not 0 <= item["slot"] < 10):  # pinned GGML_MAX_SRC
                    raise ValueError("source dispatch tensor input slots are invalid")
                slots.append(item["slot"])
                _tensor(item.get("tensor"))
            if len(set(slots)) != len(slots):
                raise ValueError("source dispatch tensor inputs are ambiguous")
        result.append({"dispatch_index": index, "first_ggml_index": first,
            "last_ggml_index": last, "first_node_ordinal": cursor,
            "node_count": end - cursor, "node_ids": ids[cursor:end],
            "nodes": tuple(nodes[cursor:end]), "source_nodes": tuple(source_nodes)})
        cursor, previous_ggml = end, last
    if cursor != len(ids):
        raise ValueError("source dispatch ownership does not cover every CUDA node")
    return tuple(result)


def iter_source_dispatches(path):
    """Yield complete source-owned calls without retaining earlier calls."""
    # Local import avoids changing the existing serving/structure dependency.
    from .cuda_graph_serving import cuda_graph_topology_descriptor

    calls, pending, labels = {}, {}, set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError("source ownership record must be an object")
            kind, call = row.get("kind"), row.get("call_id")
            if kind == "snapshot":
                if (row.get("capture_only") is not True or row.get("dry_run") is not False
                        or row.get("source_revision") != SOURCE_REVISION
                        or type(call) is not int or call in calls
                        or not isinstance(row.get("label"), str) or not row["label"]):
                    raise ValueError("source ownership requires unique capture-only snapshots")
                calls[call] = row
                pending[call] = []
            elif kind == "source_dispatch":
                if type(call) is not int or call not in pending:
                    raise ValueError("source dispatch has no active snapshot")
                pending[call].append(row)
            elif kind == "cuda_structure":
                if (type(call) is not int or call not in pending or row.get("capture_only") is not True
                        or row.get("source_revision") != SOURCE_REVISION):
                    raise ValueError("CUDA structure has no matching source snapshot")
                snapshot = calls[call]
                label = snapshot["label"]
                if label in labels:
                    raise ValueError("source dispatch labels are ambiguous")
                groups = bind_source_dispatches(cuda_graph_topology_descriptor(row),
                    pending.pop(call), ggml_node_count=snapshot["n_nodes"])
                labels.add(label)
                yield label, groups
    if pending or not labels:
        raise ValueError("source dispatch capture is incomplete")


def load_source_dispatches(path):
    """Read all capture labels with mandatory complete dispatch ownership."""
    result = dict(iter_source_dispatches(path))
    return result
