"""Predeclare and run independent queue-model training/holdout experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import tempfile
import threading
import time

import psutil

from run_cuda_graph_runtime_microbench import ROOT, complete_identity
from run_cuda_graph_launch_gap_microbench import _nonnegative, sample_intervals, stats

SOURCE = ROOT / "tools" / "cuda_dispatch_queue_microbench.cu"


def create_plan():
    cases = []
    def add(name, split, body, nodes=128, payload=0, blocks=1, threads=32, observe=True):
        bodies = body if isinstance(body, list) else [body] * nodes
        cases.append({"case_id": name, "split": split, "node_count": nodes, "extra_parameter_bytes": payload,
                      "blocks": blocks, "threads": threads, "observe_host": observe, "requested_bodies_ns": bodies})
    for body in (0, 125, 250, 375, 625, 750, 875, 1000, 1125, 1250, 1750, 2000, 4000, 5000):
        add(f"threshold_train_{body}", "train_threshold", body)
    for payload in (64, 256):
        for body in (0, 1000):
            add(f"payload_train_{payload}_{body}", "train_payload", body, payload=payload)
    for body in (500, 1500, 3000, 7000):
        add(f"threshold_holdout_{body}", "holdout_threshold", body)
    for nodes in (33, 255):
        add(f"nodes_holdout_{nodes}", "holdout_nodes", 750, nodes=nodes)
    add("geometry_holdout", "holdout_geometry", 750, blocks=32, threads=128)
    for body in (0, 1000):
        add(f"payload_holdout_1024_{body}", "holdout_payload", body, payload=1024)
    add("mixed_alternating_holdout", "holdout_mixed_body", [250, 7000] * 64)
    randomizer = random.Random(0xC0DE)
    add("mixed_shuffled_holdout", "holdout_mixed_body", [randomizer.choice((125, 375, 625, 1125, 1750, 4000)) for _ in range(128)])
    add("observer_off_payload_0", "observer_control", 0, observe=False)
    add("observer_off_payload_256", "observer_control", 0, payload=256, observe=False)
    order = list(range(len(cases)))
    random.Random(8173).shuffle(order)
    return {"schema": "heterollm.cuda-queue-probe-plan/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "target_llm_latency_used": False, "repetitions": 21, "warmups": 2,
            "mode": "ordinary_default_dependency", "gate": "device_resident_system_scope_atomic",
            "measurement_boundary": "per_call_CPU_enqueue_and_GPU_effective_body_envelopes_separate_clocks",
            "training_policy": "Fit only train_threshold/train_payload; never tune against holdout or model latency.",
            "rejection_conditions": ["incomplete or unordered CPU/GPU timestamps", "gate release precedes any API return",
                                     "unbounded threshold parameter interval", "unmodeled parameter-dependent pulse period or phase",
                                     "material observer-on/off change", "holdout failure is retained, never fixed by relabeling holdout"],
            "cases": cases, "execution_order": order}


def summarize(raw, case):
    if raw.get("schema") != "heterollm.cuda-dispatch-queue/v1" or raw.get("target_llm_latency_used") is not False:
        raise ValueError("not an independent queue probe")
    for key in ("node_count", "blocks", "threads", "extra_parameter_bytes", "requested_bodies_ns"):
        if raw.get(key) != case[key]:
            raise ValueError(f"queue probe differs from declared plan: {key}")
    if raw.get("host_observation") != case["observe_host"] or raw.get("cuda_parameter_extent_bytes") != 32 + case["extra_parameter_bytes"]:
        raise ValueError("CPU observer or actual CUDA parameter extent differs from plan")
    values, gaps, bodies, host_durations = [], [], [], []
    count = raw["node_count"]
    for sample in raw.get("samples", []):
        intervals = sample_intervals(sample, count)
        values.append(intervals["span_ns"]); gaps.append(intervals["uncovered_gap_ns"])
        bodies.extend(end - begin for begin, end in zip(sample["node_begin_ns"], sample["node_end_ns"]))
        _nonnegative(sample.get("device_event_ns"), "queue event duration")
        begin, end = sample.get("host_enqueue_begin_ns"), sample.get("host_enqueue_end_ns")
        if not case["observe_host"]:
            if begin != [] or end != [] or sample.get("host_release_ns") != 0:
                raise ValueError("unobserved queue control contains fabricated host timing")
            continue
        if not isinstance(begin, list) or not isinstance(end, list) or len(begin) != count or len(end) != count:
            raise ValueError("CPU enqueue timing directory is incomplete")
        for value in [*begin, *end, sample.get("host_release_ns")]:
            _nonnegative(value, "CPU queue timestamp")
        if begin[0] != 0 or any(right < left for left, right in zip(begin, end)) or any(begin[i] < end[i - 1] for i in range(1, count)):
            raise ValueError("CPU API timing is inverted or overlaps serialized enqueue")
        if sample["host_release_ns"] < end[-1]:
            raise ValueError("GPU gate released before every CPU submission returned")
        host_durations.append(end[-1])
    if len(values) < 3:
        raise ValueError("too few queue probe repetitions")
    return {"device_span": stats(values), "uncovered_gap": stats(gaps), "effective_body": stats(bodies),
            "host_submission_span": stats(host_durations) if host_durations else None, "prediction_qualified": False}


def build_probe(directory, nvcc, vsdevcmd):
    directory = Path(directory)
    exe = directory / "cuda_dispatch_queue_microbench.exe"
    script = directory / "build_queue.cmd"
    script.write_text(f'@call "{vsdevcmd}" -arch=amd64 -host_arch=amd64 >nul\r\n'
                      f'@"{nvcc}" -O2 -std=c++17 -arch=sm_120 "{SOURCE}" -o "{exe}"\r\n', encoding="utf-8")
    try:
        result = subprocess.run(["cmd.exe", "/d", "/c", str(script)], cwd=directory, text=True,
                                capture_output=True, encoding="utf-8", errors="replace")
    finally:
        script.unlink(missing_ok=True)
    if result.returncode:
        raise RuntimeError(f"queue probe compilation failed:\n{result.stdout}\n{result.stderr}")
    return exe


def execute_observed(command):
    """CPU monitoring is outside the probe; it never changes cost parameters."""
    samples = []
    stop = threading.Event()
    def monitor():
        psutil.cpu_percent(interval=None)
        while not stop.wait(0.05):
            samples.append({"monotonic_seconds": time.monotonic(), "cpu_percent": psutil.cpu_percent(interval=None),
                            "available_memory_bytes": psutil.virtual_memory().available})
    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    try:
        result = subprocess.run(command, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=120)
    finally:
        stop.set()
        worker.join()
    return result, {"logical_cpu_count": psutil.cpu_count(), "cpu_samples": samples,
                    "quiet_environment_qualified": False,
                    "note": "CPU contention is recorded independently; quiet-environment qualification is not assumed by the measurement tool."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--nvcc", default=os.environ.get("CUDA_PATH", r"E:\cuda") + r"\bin\nvcc.exe")
    parser.add_argument("--vsdevcmd", default=r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    plan_path = args.directory / "experiment_plan.json"
    if args.prepare_only:
        if plan_path.exists():
            raise ValueError("declared queue experiment plan already exists; do not overwrite")
        plan_path.write_text(json.dumps(create_plan(), indent=2) + "\n", encoding="utf-8")
        print(f"declared plan before measurement: {plan_path.resolve()}")
        return
    if not plan_path.exists():
        raise ValueError("prepare the train/holdout plan before running measurements")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "heterollm.cuda-queue-probe-plan/v1" or plan.get("target_llm_latency_used") is not False:
        raise ValueError("invalid queue probe plan")
    results = []; identity = None
    started = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="heterollm-queue-probe-") as directory:
        executable = build_probe(directory, args.nvcc, args.vsdevcmd)
        for index in plan["execution_order"]:
            case = plan["cases"][index]
            output = args.directory / (case["case_id"] + ".json.gz")
            if output.exists():
                raise ValueError(f"refusing to overwrite prior queue measurements: {output}")
            command = [str(executable), "--nodes", str(case["node_count"]), "--blocks", str(case["blocks"]),
                       "--threads", str(case["threads"]), "--payload-bytes", str(case["extra_parameter_bytes"]),
                       "--bodies-ns", ",".join(map(str, case["requested_bodies_ns"])),
                       "--observe-host", str(int(case["observe_host"])), "--repetitions", str(plan["repetitions"])]
            result, environment = execute_observed(command)
            if result.returncode:
                raise RuntimeError(f"queue probe {case['case_id']} failed: {result.stderr}")
            raw = json.loads(result.stdout)
            raw["measurement_environment"] = environment
            compact = summarize(raw, case)
            current = {key: raw[key] for key in ("hardware_id", "architecture", "driver_version", "runtime_version")}
            if identity is not None and identity != current:
                raise ValueError("queue probe identity changed")
            identity = current
            with gzip.open(output, "wt", encoding="utf-8", compresslevel=9) as stream:
                json.dump(raw, stream)
            results.append({"case_id": case["case_id"], "split": case["split"], "source": output.name, "summary": compact,
                            "measurement_environment": environment})
            print(f"measured {case['case_id']}", flush=True)
    complete_identity(identity)
    summary = {**identity, "schema": "heterollm.cuda-queue-probe-suite/v1", "target_llm_latency_used": False,
               "prediction_qualified": False, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "cases": results}
    (args.directory / "measurement_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(results)} planned queue experiments")


if __name__ == "__main__":
    main()
