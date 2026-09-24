#!/usr/bin/env python3
"""Phase-1 host runner: exact bounded cells with memory-aware scheduling.

This runner deliberately does not change a scenario.  It reuses the existing
matrix ``run_cell`` and only changes how a cell is hosted:

* one cell per child process, so Python task graphs are released at exit;
* exact LRU planner caches with bounded retention;
* weighted scheduling so L5/L6/L8 do not multiply host memory.

The formal matrix source identity remains the matrix runner's identity; this
file is an execution wrapper, not a simulator or an extrapolator.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "tools" / "multiworkload_architecture_matrix.py"
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools import multiworkload_architecture_matrix as matrix
from heterollm_sim.planner import TopologyAwareBatchCostProvider


EXACT_CACHE_POLICY = {
    "template": 8,
    "leaf": 256,
    "metadata_source": 256,
    "artifact_workload": 256,
}


# Weight is a host-memory budget, not a simulated hardware quantity.
LOAD_WEIGHT = {
    "L1": 1,
    "L2": 1,
    "L3": 2,
    "L4": 3,
    "L5": 8,
    "L6": 8,
    "L7": 3,
    "L8": 6,
}


def _load_prefix(load_id: str) -> str:
    return str(load_id).split("_", 1)[0]


def _weight(load_id: str) -> int:
    try:
        return LOAD_WEIGHT[_load_prefix(load_id)]
    except KeyError as exc:
        raise ValueError(f"unknown workload weight: {load_id}") from exc


def _manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise ValueError(f"missing formal manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("manifest must be an object")
    if value.get("source") != matrix.source_identity():
        raise ValueError(
            "formal source identity changed; use a new matrix directory instead of stale cells"
        )
    return value


def _acquire_lock(run_dir: Path) -> Path:
    lock = run_dir / ".phase1.lock"
    payload = f"pid={os.getpid()}\n"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"phase-1 runner already owns {lock}") from exc
    try:
        os.write(fd, payload.encode("ascii"))
    finally:
        os.close(fd)
    return lock


def _release_lock(lock: Path) -> None:
    try:
        lock.unlink()
    except FileNotFoundError:
        pass


def _bounded_worker(argv: argparse.Namespace) -> int:
    """Run exactly one formal cell with bounded *exact* caches."""

    run_dir = argv.output.resolve()
    manifest = _manifest(run_dir)
    workloads = manifest["workloads"]
    load = next(
        (tuple(item) for item in workloads if item[0] == argv.load_id),
        None,
    )
    if load is None:
        raise ValueError(f"workload is not declared by manifest: {argv.load_id}")

    original_run_scenario = matrix.q.run_scenario

    def apply_context_limits(lowerer):
        context = getattr(lowerer, "_compilation_context", None)
        if context is not None:
            context.metadata_source_cache_entries = EXACT_CACHE_POLICY[
                "metadata_source"
            ]
            context.artifact_workload_cache_entries = EXACT_CACHE_POLICY[
                "artifact_workload"
            ]

    def bounded_run_scenario(scenario, *, retention_policy):
        # This is still exact: eviction causes recompilation, never interpolation.
        lowerer = TopologyAwareBatchCostProvider(
            scenario,
            template_cache_entries=argv.template_cache_entries,
            leaf_cache_entries=argv.leaf_cache_entries,
        )
        apply_context_limits(lowerer)
        rebind = getattr(lowerer, "_rebind_control_plane_successor", None)
        if rebind is not None:
            def bounded_rebind(source, mapped_scenario):
                rebind(source, mapped_scenario)
                apply_context_limits(lowerer)
            lowerer._rebind_control_plane_successor = bounded_rebind
        return original_run_scenario(
            scenario,
            retention_policy=retention_policy,
            batch_lowerer=lowerer,
        )

    # q.execute_scenario resolves run_scenario from qwen38_memory_scenario's
    # module globals.  Rebinding that function is local to this child process.
    matrix.q.run_scenario = bounded_run_scenario
    matrix.run_cell(argv.variant, load, run_dir)
    return 0


@dataclass(frozen=True)
class Job:
    variant: str
    load: Tuple[object, ...]

    @property
    def cell_id(self) -> str:
        return f"{self.load[0]}__{self.variant}"

    @property
    def weight(self) -> int:
        return _weight(str(self.load[0]))


def _jobs(manifest: dict, selected_loads: Sequence[str]) -> List[Job]:
    selected = set(selected_loads)
    cancelled = set(manifest.get("cancelled_cells", []))
    variants = list(dict.fromkeys(
        case for group in manifest["groups"] for case in group["cases"]
    ))
    jobs = []
    for load in manifest["workloads"]:
        if selected and _load_prefix(load[0]) not in selected:
            continue
        for variant in variants:
            job = Job(variant, tuple(load))
            if job.cell_id not in cancelled:
                jobs.append(job)
    # Host-only ordering: launch the heaviest cells first so the long tail is
    # not left behind after light cells occupy the available slots.  The
    # workload, scenario, and cell identity are unchanged.
    return sorted(jobs, key=lambda job: -job.weight)


def _existing(run_dir: Path, job: Job) -> bool:
    return (run_dir / "cells" / f"{job.cell_id}.json").is_file()


def _child_command(argv: argparse.Namespace, job: Job) -> List[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-case", job.variant,
        "--worker-load-id", str(job.load[0]),
        "--output", str(argv.output.resolve()),
        "--template-cache-entries", str(argv.template_cache_entries),
        "--leaf-cache-entries", str(argv.leaf_cache_entries),
    ]


def _launch(argv: argparse.Namespace, job: Job) -> subprocess.Popen[str]:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(
        _child_command(argv, job),
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        creationflags=flags,
    )


def _run_parent(argv: argparse.Namespace) -> int:
    run_dir = argv.output.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(run_dir)
    if argv.workers < 1 or argv.worker_budget < 1:
        raise ValueError("workers and worker-budget must be positive")
    if argv.template_cache_entries < 1 or argv.leaf_cache_entries < 1:
        raise ValueError("cache limits must be positive")

    jobs = [job for job in _jobs(manifest, argv.loads) if not _existing(run_dir, job)]
    if any(job.weight > argv.worker_budget for job in jobs):
        raise ValueError("worker-budget is smaller than a heavy workload weight")
    print(
        f"PHASE1 START jobs={len(jobs)} workers={argv.workers} "
        f"budget={argv.worker_budget} cache=({argv.template_cache_entries},{argv.leaf_cache_entries})",
        flush=True,
    )

    lock = _acquire_lock(run_dir)
    running: Dict[subprocess.Popen[str], Job] = {}
    active_weight = 0
    completed = 0
    try:
        pending = list(jobs)
        while pending or running:
            launched = True
            while pending and len(running) < argv.workers and launched:
                launched = False
                for index, job in enumerate(pending):
                    if active_weight + job.weight > argv.worker_budget:
                        continue
                    process = _launch(argv, job)
                    running[process] = job
                    active_weight += job.weight
                    pending.pop(index)
                    print(
                        f"START {job.cell_id} weight={job.weight} active_weight={active_weight}",
                        flush=True,
                    )
                    launched = True
                    break

            finished = []
            for process, job in list(running.items()):
                if process.poll() is None:
                    continue
                stdout, stderr = process.communicate()
                finished.append((process, job, stdout, stderr))
            for process, job, stdout, stderr in finished:
                del running[process]
                active_weight -= job.weight
                completed += 1
                cell = run_dir / "cells" / f"{job.cell_id}.json"
                status = "MISSING"
                elapsed = ""
                if cell.is_file():
                    try:
                        result = json.loads(cell.read_text(encoding="utf-8-sig"))
                        status = result.get("status", "UNKNOWN")
                        elapsed = str(result.get("elapsed_s", ""))
                    except (OSError, ValueError):
                        status = "CORRUPT"
                print(
                    f"DONE {completed}/{len(jobs)} {job.cell_id} {status} {elapsed}",
                    flush=True,
                )
                if stdout and stdout.strip():
                    print(stdout.strip(), flush=True)
                if process.returncode != 0 and stderr.strip():
                    print(stderr.strip()[-8000:], file=sys.stderr, flush=True)
            if running and not finished:
                time.sleep(argv.poll_s)

        # Keep the ordinary matrix summary format.  It is a report snapshot,
        # not a source of truth; cells/configs remain authoritative.
        matrix.summarize(
            run_dir,
            manifest["groups"],
            [tuple(item) for item in manifest["workloads"]],
        )
        print("PHASE1 DONE", flush=True)
        return 0
    finally:
        for process in running:
            if process.poll() is None:
                process.kill()
        _release_lock(lock)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loads", nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-budget", type=int, default=8)
    parser.add_argument("--template-cache-entries", type=int, default=8)
    parser.add_argument("--leaf-cache-entries", type=int, default=256)
    parser.add_argument("--poll-s", type=float, default=0.5)
    parser.add_argument("--worker-case", dest="variant")
    parser.add_argument("--worker-load-id", dest="load_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.variant or args.load_id:
            if not args.variant or not args.load_id:
                parser.error("worker mode requires --worker-case and --worker-load-id")
            return _bounded_worker(args)
        return _run_parent(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"phase1 runner failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
