"""Analytical bandwidth envelopes and DQ conservation for the HBM core.

The HBM model has no vendor AC table, so absolute timing cannot be validated
against a datasheet.  What can be validated without one is narrower and still
useful:

* the measured service rate never exceeds the interface the configuration
  itself declares;
* a sequential stream converges onto that interface as the request grows,
  so the steady-state burst rate is not accidentally throttled;
* every transferred byte is billed exactly once, on a lane calendar that
  never overlaps, which is the half-duplex DQ constraint the model claims.

The ceilings are derived from the same DramConfig the core executes.  This is
an internal-consistency check, not an external calibration, and the report
says so.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest, Operation
from heterollm_sim.physical_presets import resolve_physical_config

GIB = 1024 ** 3
MIB = 1024 ** 2
# One burst is the model's transfer quantum; a sweep in bursts keeps every
# point comparable across organizations.
SEQUENTIAL_BURSTS = (32, 256, 2048, 16384, 131072, 1048576)
RANDOM_BURSTS = (256, 2048, 16384, 65536)
CONSERVATION_BURSTS = 4096
EPSILON_NS = 1e-9


def reference_config(capacity_bytes: int = 512 * GIB, *, read_gb_s: float = 1600.,
                     write_gb_s: float = 1600., stacks: int | None = None,
                     profile_id: str = "hbm3e_reference", refresh: bool = True):
    overrides = {"capacity_bytes": capacity_bytes,
                 "read_bandwidth_gb_s": read_gb_s, "write_bandwidth_gb_s": write_gb_s,
                 "dies_per_stack": 16, "die_density_gbits": 32}
    if stacks is not None:
        overrides["stacks"] = stacks
    if not refresh:
        # Must go through the config entry: __post_init__ re-derives
        # refresh_policy, so dataclasses.replace cannot switch it off.
        overrides["refresh_policy"] = "disabled"
    return resolve_physical_config(profile_id, overrides=overrides)


def analytic_envelope(config) -> dict[str, Any]:
    """Ceilings implied by the configuration, not by a measured device.

    ``effective_lane_bandwidth_gb_s`` paces a burst's occupancy on one lane, so
    ``lanes * burst / (burst / effective) `` is the physical interface.  The
    directional caps pace burst starts separately, which is why the model
    permits interleaved directions to exceed either cap but never the
    interface.
    """
    lanes = config.lane_count
    burst = config.burst_bytes
    lane_gb_s = config.effective_lane_bandwidth_gb_s
    burst_ns = burst / lane_gb_s
    physical = lanes * lane_gb_s
    return {
        "lanes": lanes,
        "burst_bytes": burst,
        "lane_bandwidth_gb_s": lane_gb_s,
        "burst_occupancy_ns": burst_ns,
        "physical_interface_gb_s": physical,
        "declared_read_gb_s": config.read_bandwidth_gb_s,
        "declared_write_gb_s": config.write_bandwidth_gb_s,
        "sequential_read_gb_s": min(config.read_bandwidth_gb_s, physical),
        "sequential_write_gb_s": min(config.write_bandwidth_gb_s, physical),
        "identities": {
            "physical_equals_lanes_times_lane_rate":
                abs(physical - lanes * lane_gb_s) <= 1e-9,
            "burst_occupancy_matches_burst_interval":
                abs(burst_ns - config.burst_interval_ns) <= 1e-9,
            "directional_caps_do_not_exceed_interface":
                config.read_bandwidth_gb_s <= physical + 1e-9
                and config.write_bandwidth_gb_s <= physical + 1e-9,
        },
    }


def _slowest_or_none(values):
    return max(values) if values else None


def without_refresh(config):
    """The same organization with periodic refresh switched off.

    Refresh is a real cost that grows with run length, so a refreshed stream is
    not expected to be monotone in size.  Comparing the two isolates that cost
    from the steady-state burst rate.
    """
    return replace(config, refresh_policy="disabled")


def sequential_sweep(config, bursts=SEQUENTIAL_BURSTS) -> list[dict[str, Any]]:
    """One contiguous stream per point; the rate should converge upward.

    ``capture_details`` is off so the numeric kernel serves the request and
    larger sizes stay cheap.  Nothing here reads per-burst intervals.
    """
    ceiling = analytic_envelope(config)
    rows = []
    for count in bursts:
        size = count * config.burst_bytes
        core = DramCore(config, ResourceTimeline(), capture_details=False)
        result = core.submit(AccessRequest(f"seq-{count}", Operation.READ, 0, size))
        duration = result.completion_ns - result.arrival_ns
        measured = result.logical_bytes / duration if duration > 0 else 0.0
        rows.append({
            "id": f"sequential-read-{count}bursts",
            "kind": "sequential_read",
            "bursts": count,
            "bytes": result.logical_bytes,
            "completion_ns": result.completion_ns,
            "measured_gb_s": measured,
            "ceiling_gb_s": ceiling["sequential_read_gb_s"],
            "fraction_of_ceiling": measured / ceiling["sequential_read_gb_s"],
            "burst_count": result.counters.get("burst_count"),
            "row_hits": result.counters.get("row_hits"),
            "row_misses": result.counters.get("row_misses"),
            "row_conflicts": result.counters.get("row_conflicts"),
        })
    return rows


def sequential_write_sweep(config, bursts=(16384,)) -> list[dict[str, Any]]:
    ceiling = analytic_envelope(config)
    rows = []
    for count in bursts:
        size = count * config.burst_bytes
        core = DramCore(config, ResourceTimeline(), capture_details=False)
        result = core.submit(AccessRequest(f"seqw-{count}", Operation.WRITE, 0, size))
        duration = result.completion_ns - result.arrival_ns
        measured = result.logical_bytes / duration if duration > 0 else 0.0
        rows.append({
            "id": f"sequential-write-{count}bursts",
            "kind": "sequential_write",
            "bursts": count,
            "bytes": result.logical_bytes,
            "completion_ns": result.completion_ns,
            "measured_gb_s": measured,
            "ceiling_gb_s": ceiling["sequential_write_gb_s"],
            "fraction_of_ceiling": measured / ceiling["sequential_write_gb_s"],
        })
    return rows


def random_sweep(config, bursts=RANDOM_BURSTS, *, seed: int = 20261109) -> list[dict[str, Any]]:
    """Independent bursts scattered over the whole address range.

    One burst per request keeps the comparison to the sequential sweep clean:
    any gap is row locality, not request size.
    """
    ceiling = analytic_envelope(config)
    rng = random.Random(seed)
    burst = config.burst_bytes
    span = config.capacity_bytes // burst
    requests = [AccessRequest(f"rand-{index}", Operation.READ, rng.randrange(span) * burst, burst)
                for index in range(max(bursts))]
    rows = []
    for count in bursts:
        core = DramCore(config, ResourceTimeline(), capture_details=False)
        results = core.submit_many(requests[:count])
        arrival = min(result.arrival_ns for result in results)
        completion = max(result.completion_ns for result in results)
        total = sum(result.logical_bytes for result in results)
        duration = completion - arrival
        measured = total / duration if duration > 0 else 0.0
        rows.append({
            "id": f"random-read-{count}bursts",
            "kind": "random_read",
            "bursts": count,
            "bytes": total,
            "window_ns": duration,
            "measured_gb_s": measured,
            "ceiling_gb_s": ceiling["sequential_read_gb_s"],
            "fraction_of_ceiling": measured / ceiling["sequential_read_gb_s"],
        })
    return rows


def dq_conservation(config, *, bursts: int = CONSERVATION_BURSTS, seed: int = 20261109) -> dict[str, Any]:
    """Bill every transferred byte once on a non-overlapping lane calendar.

    Interval detail requires the reference controller, so this point stays
    small.  Half-duplex is checked as "no two intervals on one lane overlap",
    which holds regardless of direction; the mixed run then confirms both
    directions still appear.
    """
    rng = random.Random(seed)
    burst = config.burst_bytes
    span = config.capacity_bytes // burst
    read_addresses = [rng.randrange(span) * burst for _ in range(bursts)]
    # Writes go to a disjoint address band so read/write ordering by address
    # never rewrites the measurement into a dependency test.
    write_addresses = [rng.randrange(span) * burst for _ in range(bursts)]
    used = set(read_addresses)
    write_addresses = [address for address in write_addresses if address not in used][:bursts]
    requests = ([AccessRequest(f"mix-r{index}", Operation.READ, address, burst)
                 for index, address in enumerate(read_addresses)]
                + [AccessRequest(f"mix-w{index}", Operation.WRITE, address, burst)
                   for index, address in enumerate(write_addresses)])

    timeline = ResourceTimeline()
    timeline.metrics_snapshot(max_intervals=8 * bursts + 1024)
    core = DramCore(config, timeline, capture_details=True)
    results = core.submit_many(requests)

    prefix = str(config.metadata.get("data_resource_prefix", "dram:data")) + ":"
    lanes = {resource: intervals for resource, intervals in timeline._intervals.items()
             if resource.startswith(prefix) and intervals}
    total_physical = sum(result.transfer_bytes for result in results)
    total_duration_ns = 0.0
    overlaps = []
    for resource, intervals in lanes.items():
        ordered = sorted(intervals)
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] < previous[1] - EPSILON_NS:
                overlaps.append({"resource": resource, "previous": previous, "current": current})
        total_duration_ns += sum(end - start for start, end in ordered)

    payload_bytes = sum(bytes_moved for payloads in timeline._interval_payloads.values()
                        for _, _, bytes_moved in payloads)
    lane_count = len(lanes)
    window_start = min((intervals[0][0] for intervals in lanes.values()), default=0.0)
    window_end = max((intervals[-1][1] for intervals in lanes.values()), default=0.0)
    window = window_end - window_start
    expected_duration_ns = total_physical / analytic_envelope(config)["lane_bandwidth_gb_s"]
    return {
        "id": "dq_conservation",
        "passed": (not overlaps and lane_count > 0
                   and payload_bytes == total_physical
                   and abs(total_duration_ns - expected_duration_ns) <= 1e-6 * max(expected_duration_ns, 1.0)),
        "metrics": {
            "requests": len(requests),
            "reads": len(read_addresses),
            "writes": len(write_addresses),
            "bursts": len(requests),
            "lanes_touched": lane_count,
            "lanes_total": config.lane_count,
            "physical_bytes": total_physical,
            "payload_bytes": payload_bytes,
            "lane_busy_ns": total_duration_ns,
            "expected_lane_busy_ns": expected_duration_ns,
            "overlap_count": len(overlaps),
            "window_ns": window,
            "aggregate_gb_s": total_physical / window if window > 0 else 0.0,
            "max_lane_utilization": (
                _slowest_or_none([sum(end - start for start, end in sorted(intervals))
                                  / max(sorted(intervals)[-1][1] - sorted(intervals)[0][0], EPSILON_NS)
                                  for intervals in lanes.values()])),
            "mixed_directions": len(read_addresses) > 0 and len(write_addresses) > 0,
        },
        "overlaps": overlaps[:16],
        "reason": ("Each 32 B burst occupies its lane for burst/effective_lane_bandwidth; "
                   "the lane calendar must serialise every burst, so an overlap would mean "
                   "one lane served two bursts at once."),
    }


def mixed_duplex_probe(config, *, bursts: int = 16384, seed: int = 20261109) -> dict[str, Any]:
    """Interleaved directions may exceed one cap but never the physical link."""
    ceiling = analytic_envelope(config)
    rng = random.Random(seed)
    burst = config.burst_bytes
    span = config.capacity_bytes // burst
    requests = []
    for index in range(bursts):
        operation = Operation.READ if index % 2 == 0 else Operation.WRITE
        requests.append(AccessRequest(f"duplex-{index}", operation,
                                      rng.randrange(span) * burst, burst))
    core = DramCore(config, ResourceTimeline(), capture_details=False)
    results = core.submit_many(requests)
    arrival = min(result.arrival_ns for result in results)
    completion = max(result.completion_ns for result in results)
    total = sum(result.logical_bytes for result in results)
    window = completion - arrival
    measured = total / window if window > 0 else 0.0
    limit = ceiling["declared_read_gb_s"] + ceiling["declared_write_gb_s"]
    return {
        "id": "mixed_duplex",
        "bursts": bursts,
        "bytes": total,
        "window_ns": window,
        "measured_gb_s": measured,
        "sum_of_directional_caps_gb_s": limit,
        "physical_interface_gb_s": ceiling["physical_interface_gb_s"],
        "bounded_by_sum_of_caps": measured <= limit * (1 + 1e-9),
        "bounded_by_physical": measured <= ceiling["physical_interface_gb_s"] * (1 + 1e-9),
    }


def refresh_overhead(config, *, bursts: int = 1048576) -> dict[str, Any]:
    """Steady-state cost of periodic refresh at a long-running point."""
    refreshed = DramCore(config, ResourceTimeline(), capture_details=False).submit(
        AccessRequest("refresh-on", Operation.READ, 0, bursts * config.burst_bytes))
    plain = reference_config(
        capacity_bytes=config.capacity_bytes, read_gb_s=config.read_bandwidth_gb_s,
        write_gb_s=config.write_bandwidth_gb_s, stacks=config.stacks,
        profile_id="hbm4_reference" if config.generation == "HBM4" else "hbm3e_reference",
        refresh=False)
    baseline = DramCore(plain, ResourceTimeline(), capture_details=False).submit(
        AccessRequest("refresh-off", Operation.READ, 0, bursts * config.burst_bytes))
    refreshed_ns = refreshed.completion_ns - refreshed.arrival_ns
    baseline_ns = baseline.completion_ns - baseline.arrival_ns
    return {
        "bursts": bursts,
        "refreshed_ns": refreshed_ns,
        "no_refresh_ns": baseline_ns,
        "refresh_commands": refreshed.counters.get("refresh_commands"),
        "overhead_fraction": refreshed_ns / baseline_ns - 1.0 if baseline_ns else None,
        "note": ("periodic refresh is a real cost that grows with run length, so a refreshed "
                 "stream is not required to be monotone in size; the no-refresh run is"),
    }


def run(output_dir: Path, *, variants: bool = True) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "parameter_basis": "configuration-derived ceilings, not vendor calibration",
        "vendor_calibrated": False,
        "validation_scope": ("internal consistency: interface ceilings, sequential convergence, "
                             "byte-once billing, half-duplex lane serialisation"),
        "variants": {},
    }
    configs = {"hbm3e_reference": reference_config()}
    if variants:
        configs["hbm4_reference"] = reference_config(profile_id="hbm4_reference")
        # Raise the directional caps onto the physical link so the array, not
        # the declared cap, becomes the binding constraint.
        caps = analytic_envelope(reference_config())
        configs["hbm3e_uncapped"] = reference_config(
            read_gb_s=caps["physical_interface_gb_s"],
            write_gb_s=caps["physical_interface_gb_s"])

    for name, config in configs.items():
        entry: dict[str, Any] = {"envelope": analytic_envelope(config)}
        entry["sequential"] = sequential_sweep(config)
        entry["sequential_write"] = sequential_write_sweep(config)
        entry["random"] = random_sweep(config)
        entry["duplex"] = mixed_duplex_probe(config)
        if config.refresh_policy != "disabled":
            # The monotone check needs the array alone; refresh is measured
            # separately so a legitimate cost is not reported as a defect.
            plain = reference_config(
                capacity_bytes=config.capacity_bytes,
                read_gb_s=config.read_bandwidth_gb_s,
                write_gb_s=config.write_bandwidth_gb_s, stacks=config.stacks,
                profile_id=("hbm4_reference" if config.generation == "HBM4"
                            else "hbm3e_reference"),
                refresh=False)
            entry["sequential_no_refresh"] = sequential_sweep(plain)
            entry["refresh_overhead"] = refresh_overhead(config)
        if name in ("hbm3e_reference", "hbm4_reference"):
            entry["dq_conservation"] = dq_conservation(config)
        report["variants"][name] = entry
        print(json.dumps({"variant": name,
                          "sequential_peak_fraction": max(row["fraction_of_ceiling"]
                                                          for row in entry["sequential"]),
                          "random_peak_fraction": max(row["fraction_of_ceiling"]
                                                      for row in entry["random"]),
                          "refresh_overhead": entry.get("refresh_overhead", {}).get(
                              "overhead_fraction"),
                          "dq_passed": entry.get("dq_conservation", {}).get("passed")}),
              flush=True)

    report["checks"] = summarise(report)
    (output_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def summarise(report: dict[str, Any]) -> dict[str, Any]:
    """Turn the measurements into pass/fail statements a reader can act on.

    Monotonicity is judged on the no-refresh stream.  A refreshed stream gains
    refresh work with run length, so requiring it to be monotone would flag a
    modelled cost as a defect.
    """
    checks: dict[str, Any] = {}
    for name, entry in report["variants"].items():
        envelope = entry["envelope"]
        sequential = entry["sequential"]
        plain = entry.get("sequential_no_refresh")
        checks[name] = {
            "identities_hold": all(envelope["identities"].values()),
            "sequential_never_exceeds_ceiling":
                all(row["fraction_of_ceiling"] <= 1.0 + 1e-9 for row in sequential),
            "sequential_converges":
                sequential[-1]["fraction_of_ceiling"] >= 0.95,
            "random_never_exceeds_ceiling":
                all(row["fraction_of_ceiling"] <= 1.0 + 1e-9 for row in entry["random"]),
            "random_below_sequential":
                max(row["fraction_of_ceiling"] for row in entry["random"])
                <= max(row["fraction_of_ceiling"] for row in sequential),
            "duplex_within_physical": entry["duplex"]["bounded_by_physical"],
        }
        if plain is not None:
            checks[name]["no_refresh_monotone_in_size"] = all(
                left["fraction_of_ceiling"] <= right["fraction_of_ceiling"] + 1e-3
                for left, right in zip(plain, plain[1:]))
        if "dq_conservation" in entry:
            checks[name]["dq_conservation"] = entry["dq_conservation"]["passed"]
    return checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path,
                        default=ROOT / "artifacts" / "hbm_bandwidth_envelope_20261109")
    parser.add_argument("--no-variants", action="store_true")
    arguments = parser.parse_args(argv)
    report = run(arguments.out, variants=not arguments.no_variants)
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2))
    return 0 if all(all(values.values()) for values in report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
