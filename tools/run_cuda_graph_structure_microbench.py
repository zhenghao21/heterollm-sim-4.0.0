"""Measure independent tiny CUDA bodies with source-compiled typed chain DAGs.

No model is loaded. Capture-only descriptors supply node types, dependency
types, and copy sizes. Real kernel functions, weights, and model latency are
never executed or fitted. The result is an exact-structure diagnostic with
reported spread, not a qualified interpolation model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

from run_cuda_graph_runtime_microbench import ROOT, build_microbenchmark, complete_identity


DEFAULT_EDGE = "0000000000000000"
PROGRAMMATIC_EDGE = "0100010000000000"
COPY_FIELDS = ("src_memory_type", "dst_memory_type", "width_bytes", "height", "depth", "src_pitch", "dst_pitch")


def typed_chain_nodes(structure):
    if structure.get("capture_only") is not True:
        raise ValueError("only capture-only structural compiler output is accepted")
    nodes = {node["id"]: node for node in structure["nodes"]}
    if len(nodes) != len(structure["nodes"]) or not nodes:
        raise ValueError("empty or duplicate CUDA node identity")
    if len(structure["edges"]) != len(nodes) - 1 or len(structure["edge_data"]) != len(structure["edges"]):
        raise ValueError("typed synthetic benchmark requires an explicit chain")
    following, incoming = {}, {}
    for (source, target), edge in zip(structure["edges"], structure["edge_data"]):
        if source not in nodes or target not in nodes or source in following or target in incoming:
            raise ValueError("typed synthetic benchmark requires an explicit chain")
        if edge not in (DEFAULT_EDGE, PROGRAMMATIC_EDGE):
            raise ValueError("unmeasured CUDA graph edge semantics")
        following[source] = target
        incoming[target] = edge
    roots = set(nodes) - set(incoming)
    if len(roots) != 1:
        raise ValueError("typed synthetic benchmark requires exactly one chain root")
    current = roots.pop()
    result = []
    while current is not None:
        node = nodes[current]
        edge = incoming.get(current, DEFAULT_EDGE)
        node_type = node["type"]
        if node_type == 0:
            result.append((0, edge, None))
        elif node_type == 1:
            copy = node.get("copy")
            if not isinstance(copy, dict) or not set(COPY_FIELDS).issubset(copy):
                raise ValueError("copy descriptor lacks independent shape/type information")
            values = tuple(copy[key] for key in COPY_FIELDS)
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError("invalid copy descriptor")
            src_type, dst_type, width, height, depth, src_pitch, dst_pitch = values
            if src_type != 2 or dst_type != 2 or height != 1 or depth != 1 or width < 1 or edge != DEFAULT_EDGE:
                raise ValueError("this measured synthetic family supports only one-dimensional device-to-device copies")
            result.append((1, edge, values))
        else:
            raise ValueError(f"unmeasured CUDA graph node type: {node_type}")
        current = following.get(current)
        if len(result) > len(nodes):
            raise ValueError("cycle in CUDA chain")
    if len(result) != len(nodes):
        raise ValueError("CUDA chain has disconnected nodes")
    return tuple(result)


def topology_key(sequence):
    # No addresses, function names, model names, or hashes enter the key.
    runs = []
    for index, (node_type, edge, copy) in enumerate(sequence):
        token = (["kernel"] if node_type == 0 else ["memcpy", *copy]) + ["root" if index == 0 else edge]
        if runs and runs[-1][0] == token:
            runs[-1][1] += 1
        else:
            runs.append([token, 1])
    if all(node_type == 0 and edge == DEFAULT_EDGE for node_type, edge, _ in sequence):
        return "chain"
    return "typed_chain/v1:" + json.dumps(runs, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=99)
    parser.add_argument("--update-pairs", type=Path, help="Source-predicted old/new typed structure pairs")
    parser.add_argument("--nvcc", default=os.environ.get("CUDA_PATH", r"E:\cuda") + r"\bin\nvcc.exe")
    parser.add_argument("--vsdevcmd", default=r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat")
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("at least three repetitions are required")
    templates = {}
    for path in args.capture_trace:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("kind") != "cuda_structure":
                continue
            sequence = typed_chain_nodes(record)
            key = (topology_key(sequence), len(sequence))
            item = templates.setdefault(key, {"sequence": sequence, "sources": []})
            origin = {"trace": str(path.resolve()), "source_revision": record["source_revision"]}
            if origin not in item["sources"]:
                item["sources"].append(origin)
    if not templates:
        raise ValueError("no source-compiled CUDA graph structures")
    results, identity, update_results, measurement_windows = [], None, [], []
    with tempfile.TemporaryDirectory(prefix="heterollm-typed-graphbench-") as directory:
        executable = build_microbenchmark(directory, args.nvcc, args.vsdevcmd)
        template_paths = {}
        for index, ((topology, size), template) in enumerate(templates.items()):
            path = Path(directory) / f"structure_{index}.txt"
            lines = ["heterollm.synthetic-chain/v1", f"typed_chain_{index}", str(size)]
            for node_type, edge, copy in template["sequence"]:
                lines.append(f"{node_type} {int(edge == PROGRAMMATIC_EDGE)} {copy[2] if copy else 0} {3 if copy else 0}")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            template_paths[(topology, size)] = path
            started = datetime.now(timezone.utc).isoformat()
            run = subprocess.run([str(executable), "--structure", str(path), "--repetitions", str(args.repetitions)],
                                 cwd=directory, text=True, capture_output=True, encoding="utf-8", errors="replace")
            measurement_windows.append({"kind": "structure", "index": index, "node_count": size,
                                        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat()})
            if run.returncode:
                raise RuntimeError(f"typed structure {index} ({size} nodes) failed: {run.stderr}")
            raw = json.loads(run.stdout)
            current_identity = {key: value for key, value in raw.items() if key != "samples"}
            if identity is None:
                identity = current_identity
            elif identity != current_identity:
                raise ValueError("hardware/runtime identity changed across synthetic graphs")
            timings = raw["samples"][0]["timings_ns"]
            # The general harness also probes a deliberately altered graph;
            # production update-failure prices require a measured old/new pair.
            timings.pop("update_failure", None)
            summaries = {}
            for phase, values in timings.items():
                center = statistics.median(values)
                summaries[phase] = {"median_ns": center,
                    "repeat_mad_ns": statistics.median(abs(value-center) for value in values),
                    "minimum_ns": min(values), "maximum_ns": max(values), "repeat_count": len(values)}
            results.append({"topology": topology, "node_count": size, "source_structures": template["sources"],
                            "timings_ns": timings, "phase_measurements": summaries})
            print(f"measured typed synthetic graph {index+1}/{len(templates)} ({size} nodes)", flush=True)
        pairs = json.loads(args.update_pairs.read_text(encoding="utf-8")) if args.update_pairs else []
        if not isinstance(pairs, list):
            raise ValueError("source-predicted update pairs must be an array")
        for pair in pairs:
            if not isinstance(pair, dict) or set(pair) != {"old", "new"}:
                raise ValueError("update pair must explicitly identify old/new structure")
            old_key = pair["old"]["topology"], pair["old"]["node_count"]
            new_key = pair["new"]["topology"], pair["new"]["node_count"]
            if old_key not in template_paths or new_key not in template_paths:
                raise ValueError("update pair not covered by captured independent templates")
            started = datetime.now(timezone.utc).isoformat()
            run = subprocess.run([str(executable), "--structure", str(template_paths[new_key]),
                "--previous-structure", str(template_paths[old_key]), "--repetitions", str(args.repetitions)],
                cwd=directory, text=True, capture_output=True, encoding="utf-8", errors="replace")
            measurement_windows.append({"kind": "update_pair", "old_node_count": old_key[1], "new_node_count": new_key[1],
                                        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat()})
            if run.returncode:
                raise RuntimeError(f"independent update pair failed: {run.stderr}")
            values = json.loads(run.stdout)["update_failure_ns"]
            center = statistics.median(values)
            update_results.append({**pair, "phase": "update_failure", "repeated_ns": values,
                "measurement": {"median_ns": center, "repeat_mad_ns": statistics.median(abs(value-center) for value in values),
                                "minimum_ns": min(values), "maximum_ns": max(values), "repeat_count": len(values)}})
            print(f"measured failed-update structural pair {old_key[1]} -> {new_key[1]}", flush=True)
    complete_identity(identity)
    identity.update(schema="heterollm.cuda-graph-structure-measurements/v1",
                    source_kind="independent_synthetic_runtime_microbenchmark",
                    target_llm_latency_used=False, prediction_qualified=False,
                    cost_scope="host_api_lifecycle_only",
                    synthetic_kernel_body="one_thread_integer_update; programmatic waits/triggers when required",
                    preserved_structure="ordered node types; all edge semantics; D2D copy bytes",
                    limitations=["Kernel functions, launch geometry, parameter footprint and shared memory are synthetic; generalization to real kernel host costs is not yet independently validated.",
                                 "CUDA event intervals include graph execution and remain diagnostics; never subtract them to replace physical GPU or DRAM costs.",
                                 "Successful update measures an identical captured graph; update_failure uses separately measured source-predicted old/new typed structures with differing node counts. Other parameter changes/failure causes are not independently validated."],
                    samples=results, update_pairs=update_results, measurement_windows=measurement_windows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(results)} independent structural diagnostic measurements to {args.output.resolve()}")


if __name__ == "__main__":
    main()
