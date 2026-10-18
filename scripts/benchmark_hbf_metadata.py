"""Small ABBA screen of exact HBF page metadata; no media simulation.

Bulk IO supplies finite page completion arrays directly. These timings measure
metadata preparation/publication only and cannot establish end-to-end speedup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim import _hbf_page_state
from heterollm_sim.hbf_staging import HbfStagingController, _use_page_helpers
from heterollm_sim.physical_presets import resolve_physical_config


PAGE = 4096


class CompletionArrays:
    """No per-page Python calls and no simulated media/fabric execution."""

    @staticmethod
    def write_flash_pages(addresses, arrivals):
        return np.asarray(arrivals, dtype=np.float64) + 75_000.

    @staticmethod
    def read_flash_pages(addresses, arrivals):
        if np.isscalar(arrivals):
            return np.full(len(addresses), float(arrivals) + 4_000.)
        return np.asarray(arrivals, dtype=np.float64) + 4_000.

    @staticmethod
    def _unexpected(*args):
        raise AssertionError("metadata screen reached a scalar/partial-page IO callback")

    read_host = write_host = read_flash = write_flash = transfer = _unexpected


def measure(config, pages, native):
    _hbf_page_state.COMPILED = native
    controller = HbfStagingController(config, 64 * 1024 ** 2)
    assert _use_page_helpers(controller, "metadata-bench", 0, 0, pages) is native
    io = CompletionArrays()
    started = time.perf_counter_ns()
    written = controller.write("metadata-bench", 0, 0, pages * PAGE, 0., io)
    write_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    read = controller.read("metadata-bench", 0, 0, pages * PAGE, 0., io)
    read_ns = time.perf_counter_ns() - started

    # Validation/copies deliberately sit outside every timed phase.
    state = controller._read_states[("metadata-bench", 0)]
    before = controller.snapshot()
    order = (tuple(controller._pages),
             tuple((block, tuple(owners.items())) for block, owners in controller._owners.items()),
             tuple(controller._buffer_pages[("metadata-bench", 0)]))
    read_state = tuple(value.copy() for value in (state.addresses, state.ready, state.valid))
    assert all(state.addresses[index] == controller._address(*page.flash)
               and state.ready[index] == page.ready_ns
               for (_, _, index), page in controller._pages.items())
    started = time.perf_counter_ns()
    released = controller.release("metadata-bench", 0, 0.)
    release_ns = time.perf_counter_ns() - started
    signature = (before, order, read_state, controller.snapshot(), (written, read, released))
    return {"native": native, "write_ns": write_ns, "first_read_ns": read_ns,
            "release_ns": release_ns, "sum_ns": write_ns + read_ns + release_ns}, signature


def assert_equal(reference, candidate):
    assert reference[0] == candidate[0], "controller snapshot differs before release"
    assert reference[1] == candidate[1], "page/owner/index insertion order differs"
    assert all(np.array_equal(a, b) for a, b in zip(reference[2], candidate[2])), "dense state differs"
    assert reference[3] == candidate[3], "controller snapshot differs after release"
    assert reference[4] == candidate[4], "completion differs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--output", type=Path,
        default=ROOT / "artifacts" / "fast_mode_v2_20261017" / "hbf_metadata_micro.json")
    args = parser.parse_args()
    if not _hbf_page_state.COMPILED:
        parser.error("build _hbf_page_state first; the candidate must be native")
    if args.rounds < 1 or any(count < 16 for count in args.pages):
        parser.error("rounds must be positive and each page count at least 16")
    config = resolve_physical_config("hbf_ocp_0_7")
    results = {"scope": "exact page metadata only; bulk completion arrays, no physical media execution",
        "end_to_end_speedup_measured": False,
        "timing_interference": "ongoing isolated long reference; screen only, not a clean CPU benchmark",
        "geometry": {"array_units": config.array_units, "pages_per_block": config.pages_per_block,
                     "capacity_bytes": config.effective_capacity_bytes},
        "candidate_extension": str(Path(_hbf_page_state.__file__).resolve()),
        "runs": [], "summaries": []}
    original = _hbf_page_state.COMPILED
    try:
        # Import/cache and first object-path warmup; not reported as a speedup.
        for native in (False, True):
            measure(config, 32, native)
        for pages in args.pages:
            measurements, paired = [], []
            for iteration in range(args.rounds):
                signature = None
                rows = []
                for position, native in enumerate((False, True, True, False)):
                    row, current = measure(config, pages, native)
                    if signature is None:
                        signature = current
                    else:
                        assert_equal(signature, current)
                    row.update(pages=pages, round=iteration + 1, abba_position=position + 1)
                    rows.append(row)
                    measurements.append(row)
                paired.extend([rows[0]["sum_ns"] / rows[1]["sum_ns"],
                               rows[3]["sum_ns"] / rows[2]["sum_ns"]])
            summary = {"pages": pages, "rounds": args.rounds, "all_states_exactly_equal": True,
                "paired_sum_speedups": paired, "each_pair_improved": all(value > 1 for value in paired)}
            for phase in ("write_ns", "first_read_ns", "release_ns", "sum_ns"):
                old = fmean(row[phase] for row in measurements if not row["native"])
                new = fmean(row[phase] for row in measurements if row["native"])
                summary[phase.removesuffix("_ns")] = {
                    "python_mean_ms": old / 1e6, "native_mean_ms": new / 1e6, "speedup": old / new}
            results["runs"].extend(measurements)
            results["summaries"].append(summary)
    finally:
        _hbf_page_state.COMPILED = original
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summaries"], indent=2))


if __name__ == "__main__":
    main()
