"""Independent typed CUDA chain activity, with paired CUPTI-off perturbation checks."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

from run_cuda_graph_runtime_microbench import ROOT, complete_identity
from run_cuda_graph_launch_gap_microbench import _nonnegative, sample_intervals, stats

SOURCE = ROOT / "tools" / "cuda_graph_typed_activity_microbench.cu"


def validate_nodes(nodes):
    if not isinstance(nodes, list) or len(nodes) < 2:
        raise ValueError("typed chain needs a complete node directory")
    for index, node in enumerate(nodes):
        if not isinstance(node, list) or len(node) != 3 or any(type(x) is not int for x in node):
            raise ValueError("invalid typed node descriptor")
        kind, dependency, size = node
        if kind not in (0, 1) or dependency not in (0, 1) or (index == 0 and dependency):
            raise ValueError("unsupported typed node or edge")
        if (kind == 0 and size != 0) or (kind == 1 and (dependency != 0 or size <= 0)):
            raise ValueError("unsupported typed copy boundary")


def summarize_activity(raw):
    if raw.get("schema") != "heterollm.cuda-typed-activity/v1" or raw.get("target_llm_latency_used") is not False:
        raise ValueError("not independent typed CUDA activity")
    nodes = raw.get("nodes")
    validate_nodes(nodes)
    if raw.get("node_count") != len(nodes) or type(raw.get("activity_enabled")) is not bool:
        raise ValueError("incomplete typed measurement configuration")
    groups, events = defaultdict(list), defaultdict(list)
    for sample in raw.get("samples", []):
        mode = sample.get("mode")
        if mode not in ("ordinary", "first_launch", "replay"):
            raise ValueError("unsupported typed activity mode")
        event_duration = _nonnegative(sample.get("device_event_ns"), "CUPTI control event duration")
        if event_duration == 0:
            raise ValueError("zero CUPTI control event duration")
        events[mode].append(event_duration)
        activity = sample.get("activities")
        if not raw["activity_enabled"]:
            if activity != []:
                raise ValueError("untraced control contains profiler activity")
            continue
        if not isinstance(activity, list) or len(activity) != len(nodes):
            raise ValueError("CUPTI dropped or added typed node records")
        starts, ends = [], []
        graph_nodes = set()
        for node, observed in zip(nodes, activity):
            if not isinstance(observed, list) or len(observed) != 5 or observed[0] != node[0] or observed[3] != node[2]:
                raise ValueError("CUPTI node type/bytes differ from source chain")
            if any(type(value) is not int or value < 0 for value in observed):
                raise ValueError("invalid CUPTI activity field")
            if (mode == "ordinary") != (observed[4] == 0):
                raise ValueError("Graph activity identity does not match launch mode")
            if mode != "ordinary":
                if observed[4] in graph_nodes:
                    raise ValueError("duplicate Graph activity node identity")
                graph_nodes.add(observed[4])
            starts.append(observed[1]); ends.append(observed[2])
        intervals = sample_intervals({"node_begin_ns": starts, "node_end_ns": ends}, len(nodes))
        edge_groups = defaultdict(list)
        for index, gap in enumerate(intervals["signed_edge_gaps_ns"], start=1):
            previous, current = nodes[index - 1], nodes[index]
            key = (previous[0], current[0], current[1], previous[2], current[2])
            edge_groups[key].append(gap)
        for key, gaps in edge_groups.items():
            groups[mode, key].append({"mean": statistics.mean(gaps), "overlap": sum(value < 0 for value in gaps), "count": len(gaps)})
    if set(events) != {"ordinary", "first_launch", "replay"} or len({len(values) for values in events.values()}) != 1:
        raise ValueError("unbalanced typed activity modes")
    if min(map(len, events.values())) < 3:
        raise ValueError("insufficient typed activity repetitions")
    return {
        "event_measurements": {mode: stats(values) for mode, values in sorted(events.items())},
        "edge_measurements": [{"mode": mode, "from_kind": key[0], "to_kind": key[1], "incoming_dependency": key[2],
                               "from_copy_bytes": key[3], "to_copy_bytes": key[4], "edge_count_per_repetition": values[0]["count"],
                               "mean_signed_gap": stats([value["mean"] for value in values]),
                               "overlapping_edges": sum(value["overlap"] for value in values),
                               "prediction_qualified": False,
                               "serial_additive_candidate": not any(value["overlap"] for value in values)}
                              for (mode, key), values in sorted(groups.items())],
    }


def build_probe(directory, nvcc, vsdevcmd):
    directory = Path(directory)
    cuda = Path(nvcc).resolve().parents[1]
    cupti = cuda / "extras" / "CUPTI"
    executable = directory / "cuda_graph_typed_activity_microbench.exe"
    script = directory / "build_typed_activity.cmd"
    script.write_text(f'@call "{vsdevcmd}" -arch=amd64 -host_arch=amd64 >nul\r\n'
                      f'@"{nvcc}" -O2 -std=c++17 -arch=sm_120 -I"{cupti / "include"}" "{SOURCE}" "{cupti / "lib64" / "cupti.lib"}" -o "{executable}"\r\n', encoding="utf-8")
    try:
        result = subprocess.run(["cmd.exe", "/d", "/c", str(script)], cwd=directory, text=True,
                                capture_output=True, encoding="utf-8", errors="replace")
    finally:
        script.unlink(missing_ok=True)
    if result.returncode:
        raise RuntimeError(f"typed activity probe build failed:\n{result.stdout}\n{result.stderr}")
    return executable, cupti / "lib64"


def templates():
    for edge in (0, 1):
        yield f"kernel_edge_{edge}", [[0, edge if i else 0, 0] for i in range(128)], 0
    yield "kernel_alternating", [[0, i % 2, 0] for i in range(128)], 0
    for size in (3584, 8192, 14336, 16384, 20480, 24576):
        nodes = [[0, 0, 0]]
        for _ in range(24):
            nodes.extend([[0, 1, 0], [1, 0, size], [0, 0, 0], [0, 1, 0]])
        yield f"copy_{size}", nodes, 0
    for body in (1000, 10000):
        yield f"body_{body}", [[0, 0, 0] for _ in range(128)], body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument("--build-only", type=Path)
    parser.add_argument("--nvcc", default=os.environ.get("CUDA_PATH", r"E:\cuda") + r"\bin\nvcc.exe")
    parser.add_argument("--vsdevcmd", default=r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat")
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("at least three repetitions required")
    if args.build_only:
        args.build_only.mkdir(parents=True, exist_ok=True)
        print(build_probe(args.build_only, args.nvcc, args.vsdevcmd))
        return
    if not args.output:
        parser.error("--output is required")
    results = []; identity = None
    started = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="heterollm-typed-activity-") as directory:
        executable, dll_directory = build_probe(directory, args.nvcc, args.vsdevcmd)
        env = dict(os.environ, PATH=str(dll_directory) + os.pathsep + os.environ.get("PATH", ""))
        for name, nodes, body in templates():
            structure = Path(directory) / "typed_chain.txt"
            structure.write_text("\n".join(["heterollm.synthetic-chain/v1", name, str(len(nodes)),
                                             *(f"{kind} {edge} {size} {3 if kind else 0}" for kind, edge, size in nodes)]) + "\n", encoding="utf-8")
            paired = []
            for enabled in (False, True):
                command = [str(executable), "--structure", str(structure), "--activity", str(int(enabled)),
                           "--body-ns", str(body), "--repetitions", str(args.repetitions)]
                result = subprocess.run(command, env=env, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=180)
                if result.returncode:
                    raise RuntimeError(f"typed activity {name} activity={enabled} failed: {result.stderr}")
                raw = json.loads(result.stdout); raw["summary"] = summarize_activity(raw)
                current = {key: raw[key] for key in ("hardware_id", "architecture", "driver_version", "runtime_version")}
                if identity is not None and identity != current:
                    raise ValueError("typed activity hardware/runtime changed")
                identity = current; paired.append(raw)
            overhead = {}
            for mode in ("ordinary", "first_launch", "replay"):
                plain = paired[0]["summary"]["event_measurements"][mode]["median_ns"]
                traced = paired[1]["summary"]["event_measurements"][mode]["median_ns"]
                overhead[mode] = {"untraced_ns": plain, "traced_ns": traced, "difference_percent": (traced / plain - 1) * 100}
            results.append({"name": name, "measurements": paired, "activity_perturbation": overhead})
            print(f"measured typed CUPTI/control {name}", flush=True)
    complete_identity(identity)
    output = {**identity, "schema": "heterollm.cuda-typed-activity-suite/v1", "target_llm_latency_used": False,
              "source_kind": "independent_synthetic_typed_activity_microbenchmark", "prediction_qualified": False,
              "measurement_boundary": "CUPTI_complete_kernel_and_D2D_copy_activity", "gate": "device_resident_system_scope_atomic",
              "limitations": ["CUPTI observation can perturb execution; each case has an untraced event-time control.",
                              "Whole-kernel activity includes programmatic dependency waiting; negative PDL edge gaps are preserved, never used as negative added task durations.",
                              "Copy execution bytes/time and synthetic kernel bodies remain diagnostics, never additional runtime dispatch cost.",
                              "These are independent synthetic operations, not target-model latency calibration."],
              "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "experiments": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"saved typed activity evidence to {args.output.resolve()}")


if __name__ == "__main__":
    main()
