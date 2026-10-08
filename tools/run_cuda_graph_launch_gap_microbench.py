"""Build/run independent per-node CUDA launch-gap probes without model timing.

The measured interval is between instrumented effective kernel bodies. It
includes kernel entry/exit and probe epilogue, and is not an absolute pure
hardware scheduler constant. Queue gating isolates host submission starvation.
No measured body duration is added to an existing physical kernel cost.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

from run_cuda_graph_runtime_microbench import ROOT, complete_identity

SOURCE = ROOT / "tools" / "cuda_graph_launch_gap_microbench.cu"
SCHEMA = "heterollm.cuda-launch-gap/v1"
MODES = ("ordinary", "first_launch", "uploaded_first_launch", "replay")


def _nonnegative(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {label}")
    return float(value)


def sample_intervals(sample, node_count):
    """Compute observable gaps; preserve overlap instead of pretending it is zero."""
    if type(node_count) is not int or node_count < 2:
        raise ValueError("launch-gap observation requires at least two nodes")
    starts, ends = sample.get("node_begin_ns"), sample.get("node_end_ns")
    if not isinstance(starts, list) or not isinstance(ends, list) or len(starts) != node_count or len(ends) != node_count:
        raise ValueError("incomplete node timestamp directory")
    starts = [_nonnegative(value, "node start") for value in starts]
    ends = [_nonnegative(value, "node end") for value in ends]
    if starts[0] != 0 or any(end < start for start, end in zip(starts, ends)):
        raise ValueError("inverted or non-relative node interval")
    if any(right < left for left, right in zip(starts, starts[1:])):
        raise ValueError("node start order contradicts ordered chain")
    signed_gaps = [starts[i] - ends[i - 1] for i in range(1, node_count)]
    span = max(ends) - starts[0]
    body_sum = sum(end - start for start, end in zip(starts, ends))
    # Union subtraction also works when PDL effective bodies overlap. A
    # negative signed edge is retained and disqualifies an additive edge fit.
    merged_end = ends[0]
    union = ends[0] - starts[0]
    for start, end in zip(starts[1:], ends[1:]):
        union += max(0.0, end - max(start, merged_end))
        merged_end = max(merged_end, end)
    return {
        "span_ns": span,
        "body_sum_ns": body_sum,
        "body_union_ns": union,
        "uncovered_gap_ns": span - union,
        "signed_edge_gaps_ns": signed_gaps,
        "mean_signed_edge_gap_ns": statistics.mean(signed_gaps),
        "overlapping_edge_count": sum(gap < 0 for gap in signed_gaps),
    }


def stats(values):
    if not values:
        raise ValueError("empty independent measurement")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError("non-finite independent measurement")
    center = statistics.median(values)
    return {"median_ns": center, "repeat_mad_ns": statistics.median(abs(x - center) for x in values),
            "minimum_ns": min(values), "maximum_ns": max(values), "repeat_count": len(values)}


def summarize(raw):
    if raw.get("schema") != SCHEMA or raw.get("target_llm_latency_used") is not False:
        raise ValueError("not independent CUDA launch-gap evidence")
    count = raw.get("node_count")
    groups = defaultdict(list)
    for sample in raw.get("samples", []):
        mode, gated = sample.get("mode"), sample.get("gated")
        if mode not in MODES or type(gated) is not bool:
            raise ValueError("unknown launch-gap measurement boundary")
        _nonnegative(sample.get("host_submit_ns"), "host submission time")
        groups[mode, gated].append(sample_intervals(sample, count))
    # The initial experiment predates the optional pre-upload control. Its
    # three required modes are still complete evidence; never fill missing
    # ordinary/first/replay or one missing gate boundary from another mode.
    modes = set(MODES) - {"uploaded_first_launch"}
    if any(mode == "uploaded_first_launch" for mode, _ in groups):
        modes.add("uploaded_first_launch")
    expected = {(mode, gated) for mode in modes for gated in (False, True)}
    if set(groups) != expected or len({len(values) for values in groups.values()}) != 1:
        raise ValueError("unbalanced ordinary/Graph/gated experiment")
    if min(map(len, groups.values())) < 3:
        raise ValueError("insufficient independent repetitions")
    return [{"mode": mode, "gated": gated,
             "mean_signed_edge_gap": stats([x["mean_signed_edge_gap_ns"] for x in values]),
             "uncovered_gap": stats([x["uncovered_gap_ns"] for x in values]),
             "body_union": stats([x["body_union_ns"] for x in values]),
             "device_span": stats([x["span_ns"] for x in values]),
             "overlapping_edge_count": sum(x["overlapping_edge_count"] for x in values),
             "prediction_qualified": False,
             "serial_additive_candidate": not any(x["overlapping_edge_count"] for x in values)}
            for (mode, gated), values in sorted(groups.items())]


def build_probe(directory, nvcc, vsdevcmd):
    directory = Path(directory)
    executable = directory / "cuda_graph_launch_gap_microbench.exe"
    script = directory / "build_launch_gap.cmd"
    # Fixed architecture avoids invoking device discovery as part of compilation.
    script.write_text(f'@call "{vsdevcmd}" -arch=amd64 -host_arch=amd64 >nul\r\n'
                      f'@"{nvcc}" -O2 -std=c++17 -arch=sm_120 "{SOURCE}" -o "{executable}"\r\n', encoding="utf-8")
    try:
        result = subprocess.run(["cmd.exe", "/d", "/c", str(script)], cwd=directory, text=True,
                                capture_output=True, encoding="utf-8", errors="replace")
    finally:
        script.unlink(missing_ok=True)
    if result.returncode:
        raise RuntimeError(f"independent launch probe failed to build:\n{result.stdout}\n{result.stderr}")
    return executable


def experiments():
    for edge in ("default", "programmatic"):
        for nodes, blocks, threads, body, split in (
            (16, 1, 32, 0, "train"), (128, 1, 32, 0, "train"),
            (32, 1, 32, 0, "holdout_nodes"), (256, 1, 32, 0, "holdout_nodes"),
            (128, 1, 32, 1000, "holdout_body"), (128, 1, 32, 10000, "holdout_body"),
            (128, 32, 128, 0, "holdout_geometry"),
        ):
            yield {"nodes": nodes, "blocks": blocks, "threads": threads, "body_ns": body,
                   "edge": edge, "split": split}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument("--flush-query", type=int, choices=(0, 1), default=1)
    parser.add_argument("--gate", choices=("gpu", "host"), default="gpu")
    parser.add_argument("--build-only", type=Path, help="Compile in this directory; never execute the CUDA probe")
    parser.add_argument("--nvcc", default=os.environ.get("CUDA_PATH", r"E:\cuda") + r"\bin\nvcc.exe")
    parser.add_argument("--vsdevcmd", default=r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat")
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("at least three repetitions are required")
    if args.build_only:
        args.build_only.mkdir(parents=True, exist_ok=True)
        print(build_probe(args.build_only, args.nvcc, args.vsdevcmd))
        return
    if not args.output:
        parser.error("--output is required unless --build-only is specified")
    started = datetime.now(timezone.utc).isoformat()
    results = []
    with tempfile.TemporaryDirectory(prefix="heterollm-launch-gap-") as directory:
        executable = build_probe(directory, args.nvcc, args.vsdevcmd)
        identity = None
        for experiment in experiments():
            cmd = [str(executable), "--repetitions", str(args.repetitions), "--flush-query", str(args.flush_query), "--gate", args.gate]
            for name in ("nodes", "blocks", "threads", "body_ns", "edge"):
                cmd.extend(["--" + name.replace("_", "-"), str(experiment[name])])
            result = subprocess.run(cmd, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=180)
            if result.returncode:
                raise RuntimeError(f"launch-gap experiment failed {experiment}: {result.stderr}")
            raw = json.loads(result.stdout)
            current = {key: raw[key] for key in ("hardware_id", "architecture", "driver_version", "runtime_version")}
            if identity is not None and identity != current:
                raise ValueError("device/runtime changed during launch-gap experiment")
            identity = current
            raw["split"] = experiment["split"]
            raw["summary"] = summarize(raw)
            results.append(raw)
            print(f"measured {experiment}", flush=True)
    complete_identity(identity)
    output = {**identity, "schema": "heterollm.cuda-launch-gap-suite/v1",
              "source_kind": "independent_synthetic_launch_gap_microbenchmark", "target_llm_latency_used": False,
              "prediction_qualified": False, "measurement_boundary": "effective_kernel_body_envelopes_globaltimer",
              "cost_scope": "between_node_launch_gaps_only_excludes_first_node_dispatch",
              "limitations": [
                  "Probe entry/exit and PDL dependency completion tails remain part of the observed gap.",
                  "Device-resident gated samples preload submission; this establishes host API completion, not all WDDM command-buffer residency. Host callback gated samples cannot establish device-only dispatch.",
                  "Ungated samples are separate diagnostics and must not be added to existing host costs.",
                  "Body duration, device span and D2D traffic are never added as runtime dispatch costs.",
                  "Geometry/body holdouts must qualify a launch-gap model before predictive use; these are synthetic kernels only.",
                  "First-node dispatch latency is unobserved and must not be silently inferred from inter-node gaps."],
              "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "experiments": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"saved independent launch-gap suite to {args.output.resolve()}")


if __name__ == "__main__":
    main()
