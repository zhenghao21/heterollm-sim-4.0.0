"""Physical KV hot-read, capacity-fraction and read/write traffic studies.

Hit counts come from a deterministic page-ID trace and live KV ownership.
Cold reads physically restore into HBM and return to HBF after consumption.
These are memory microbenchmarks, with no GPU token-throughput semantics.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from hbf_hbm_sessions import PAGE, TOKENS_PER_PAGE, _hardware, _executed_core_configurations
from heterollm_sim.communication import TopologyRouter
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.kv_pool import DynamicKVPool, KvPoolComponent
from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest, Operation
from heterollm_sim.physical_presets import resolve_physical_config
from heterollm_sim.nand_core import NandCore

TOTAL_CAPACITY_PAGES = 100
HOT_PAGES = 8
COLD_PAGES = 32
READ_COUNT = 100


def _delta(after: dict, before: dict) -> dict:
    return {key: value-before.get(key, 0) for key, value in after.items()
            if value != before.get(key, 0)}


def run_hot_read_trace(*, hot_reads: int, hbm_pages: int = 25) -> dict[str, Any]:
    if isinstance(hot_reads, bool) or not isinstance(hot_reads, int) or not 0 <= hot_reads <= READ_COUNT:
        raise ValueError("hot_reads must be an integer within the 100-read trace")
    if isinstance(hbm_pages, bool) or not isinstance(hbm_pages, int) or not 0 <= hbm_pages <= TOTAL_CAPACITY_PAGES:
        raise ValueError("hbm_pages must be an integer within the total capacity budget")
    hbf_pages = TOTAL_CAPACITY_PAGES-hbm_pages
    components = [KvPoolComponent("hbm", hbm_pages*PAGE, kind="hbm", active=True,
                                  read_bandwidth_gbps=4096., write_bandwidth_gbps=4096.),
                  KvPoolComponent("hbf", hbf_pages*PAGE, kind="hbf", active=False,
                                  read_bandwidth_gbps=488.*8., write_bandwidth_gbps=27.2*8.)]
    base = {
        "mode": "physical_staged_kv_read_microbenchmark",
        "hbm_fraction": hbm_pages/TOTAL_CAPACITY_PAGES,
        "requested_hot_read_fraction": hot_reads/READ_COUNT,
        "requested_read_count": READ_COUNT, "requested_hot_read_count": hot_reads,
        "page_bytes": PAGE, "total_capacity_pages": TOTAL_CAPACITY_PAGES,
        "hbm_capacity_pages": hbm_pages, "hbf_capacity_pages": hbf_pages,
        "resident_working_set_pages": HOT_PAGES+COLD_PAGES,
        "initial_hot_pages": HOT_PAGES, "initial_cold_pages": COLD_PAGES,
        "node_tps": None, "per_gpu_tps": None, "ttft_ns": None, "tpot_ns": None,
        "policy": "fixed eight-page HBM hot set; each cold read restores one page to HBM, reads it, then returns it to HBF",
    }
    # Apply the actual zero/limited HBM capacity to the production pool before
    # declaring feasibility. The first page cannot silently spill to HBF.
    probe = DynamicKVPool(components, tokens_per_page=TOKENS_PER_PAGE, page_bytes=PAGE,
                          offload_components=("hbf",))
    required_active = HOT_PAGES + (1 if hot_reads < READ_COUNT else 0)
    admitted = probe.resize_detailed("staging-probe", required_active, compatible_components=("hbm",)).success
    if not admitted or hbf_pages < COLD_PAGES:
        return {**base, "runnable": False, "status": "configuration_rejected",
                "capacity_bytes": dict(probe.capacity_bytes_by_component),
                "reason": probe.last_error or "HBF capacity is below the fixed cold working set",
                "rejection_scope": "this HBM-staging policy only; direct all-KV-HBF is a different architecture",
                "memory_service_duration_ns": None, "observed_hot_read_fraction": None,
                "executed_read_count": 0, "events": []}

    hardware = _hardware(hbm_pages=hbm_pages, hbf_pages=hbf_pages)
    hbm_config = hardware.components[0].metadata["physical_memory_config"]
    hbf_config = hardware.components[1].metadata["physical_memory_config"]
    applied_hardware = {
        "assumption_status": "explicit synthetic parameters; uncalibrated",
        "hbm": {"capacity_bytes": hbm_pages*PAGE,
                "interface_bandwidth_gb_s": hbm_config.interface_bandwidth_gb_s},
        "hbf": {key: getattr(hbf_config, key) for key in (
            "capacity_bytes", "page_bytes", "channels", "parallel_units", "host_read_bandwidth_gb_s",
            "host_write_bandwidth_gb_s", "internal_bandwidth_gb_s", "page_read_ns", "page_program_ns", "front_ns", "write_buffer_bytes")},
        "link": {"bandwidth_gb_s": hardware.links[0].bandwidth_gbps/8.,
                 "latency_ns": hardware.links[0].latency_ns},
    }
    runtime = PhysicalRuntimeContext(capture_details=False)
    router = TopologyRouter(hardware)
    now = [0.]
    executor = PhysicalKVTransferExecutor(hardware.components, runtime=runtime, topology=router,
                                          arrival_ns=lambda _: now[0], namespace="traffic-hot-read")
    pool = DynamicKVPool(components, tokens_per_page=TOKENS_PER_PAGE, page_bytes=PAGE,
                         topology=router, offload_components=("hbf",), transfer_executor=executor)
    peak = {"hbm": 0, "hbf": 0}

    def capture_peak():
        for owner, value in pool.used_bytes_by_component.items():
            peak[owner] = max(peak[owner], value)

    def move(page_id: int, target: str):
        page = pool.page(page_id)
        before = dict(pool.used_bytes_by_component)
        peak[target] = max(peak[target], before.get(target, 0)+page.bytes)
        operation = pool.restore_page if target == "hbm" else pool.offload_page
        if not operation(page_id, target):
            raise RuntimeError(pool.last_error or "physical page move failed")
        now[0] = max(now[0], float(pool.events[-1].completion_ns))
        capture_peak()

    def seed(request: str, pages: int):
        if not pool.resize_detailed(request, pages, compatible_components=("hbm",)).success:
            raise RuntimeError(pool.last_error or "initial page allocation failed")
        executor.sync_pages(pool.pages())
        capture_peak()
        for page in pool.request_pages(request):
            result = executor.write_page(page, arrival_ns=now[0])
            now[0] = float(result["completion_ns"])

    seed("hot", HOT_PAGES)
    hot_ids = [page.logical_page_id for page in pool.request_pages("hot")]
    cold_ids = []
    for index in range(COLD_PAGES):
        request = f"cold-{index}"
        seed(request, 1)
        page_id = pool.request_pages(request)[0].logical_page_id
        move(page_id, "hbf")
        cold_ids.append(page_id)

    start = now[0]
    transfer_start = len(executor.events)
    bytes_before = dict(runtime.timeline.bytes_moved)
    busy_before = dict(runtime.timeline.busy_ns)
    counters_before = dict(executor.counters)
    events = []
    hits = misses = 0
    consumed_physical_read_bytes = 0
    # Spread the exact number of hot reads through the trace with integer
    # arithmetic. The requested class is checked against actual page owner.
    for index in range(READ_COUNT):
        requested_hot = (index+1)*hot_reads//READ_COUNT > index*hot_reads//READ_COUNT
        page_id = hot_ids[hits % len(hot_ids)] if requested_hot else cold_ids[misses % len(cold_ids)]
        owner = pool.page(page_id).owner_component
        hit = owner == "hbm"
        access_start = now[0]
        if not hit:
            move(page_id, "hbm")
        result = executor.read_page(pool.page(page_id), arrival_ns=now[0])
        now[0] = float(result["completion_ns"])
        consumed_physical_read_bytes += int(result.get("physical_read_bytes", 0))
        consumed_at = now[0]
        if not hit:
            move(page_id, "hbf")
        hits += int(hit)
        misses += int(not hit)
        events.append({"sequence": index, "logical_page_id": page_id,
                       "requested_class": "hot" if requested_hot else "cold",
                       "owner_before_read": owner, "observed_hbm_hit": hit,
                       "start_ns": access_start, "read_complete_ns": consumed_at,
                       "policy_complete_ns": now[0], "owner_after_policy": pool.page(page_id).owner_component})

    end = now[0]
    duration = end-start
    transfers = executor.events[transfer_start:]
    hbf_read = sum(int(phase.get("physical_read_bytes", 0)) for event in transfers
                   if event["source_component"] == "hbf" for phase in event["phases"] if phase["kind"] == "source_read")
    hbf_write = sum(int(phase.get("physical_write_bytes", 0)) for event in transfers
                    if event["target_component"] == "hbf" for phase in event["phases"] if phase["kind"] == "target_write")
    transfer_rows = [{key: event[key] for key in ("logical_page_id", "source_component", "target_component", "bytes", "start_ns", "completion_ns")}
                     for event in transfers]
    resource_bytes = _delta(dict(runtime.timeline.bytes_moved), bytes_before)
    resource_busy = _delta(dict(runtime.timeline.busy_ns), busy_before)
    measured_counters = _delta(dict(executor.counters), counters_before)
    capacity = dict(pool.capacity_bytes_by_component)
    request_ids = ["hot", *(f"cold-{index}" for index in range(COLD_PAGES))]
    for request in request_ids:
        pool.release(request)
    executor.sync_pages(pool.pages())
    checks = {
        "fixed_capacity_applied": sum(capacity.values()) == TOTAL_CAPACITY_PAGES*PAGE and capacity["hbm"] == hbm_pages*PAGE,
        "exact_actual_hot_read_count": hits == hot_reads,
        "all_requested_reads_executed": hits+misses == READ_COUNT,
        "cold_reads_physically_restore_and_return": len(transfers) == 2*misses,
        "cold_read_byte_conservation": hbf_read == misses*PAGE and hbf_write == misses*PAGE,
        "capacity_including_target_copy": all(peak[owner] <= capacity[owner] for owner in capacity),
        "all_pages_released": not pool.pages() and not any(allocator.allocations() for allocator in runtime.allocators.values()),
    }
    return {**base, "runnable": True, "status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "applied_hardware": applied_hardware,
            "executed_physical_configurations": _executed_core_configurations(runtime),
            "capacity_bytes": capacity, "peak_bytes_including_migration_copies": peak,
            "initialization_duration_ns": start,
            "initialization_scope": "40 actual initial page writes and 32 cold-page offloads; excluded from the measured read window",
            "measurement_window": {"start_ns": start, "end_ns": end, "duration_ns": duration},
            "memory_service_duration_ns": duration,
            "executed_read_count": hits+misses, "observed_hot_read_count": hits,
            "observed_cold_read_count": misses, "observed_hot_read_fraction": hits/READ_COUNT,
            "useful_logical_read_bytes": READ_COUNT*PAGE,
            "effective_useful_read_gb_s": READ_COUNT*PAGE/duration,
            "hbf_physical_read_bytes": hbf_read, "hbf_physical_write_bytes": hbf_write,
            "hbm_consumption_physical_read_bytes": consumed_physical_read_bytes,
            "hbf_media_read_gb_s": hbf_read/duration, "hbf_media_write_gb_s": hbf_write/duration,
            "migration_bytes": sum(event["bytes"] for event in transfers),
            "executor_counters": measured_counters, "resource_bytes": resource_bytes,
            "resource_busy_ns": resource_busy, "events": events, "transfers": transfer_rows}


def run_read_write_trace(reads_per_write: int) -> dict[str, Any]:
    if isinstance(reads_per_write, bool) or not isinstance(reads_per_write, int) or reads_per_write < 1:
        raise ValueError("reads_per_write must be a positive integer")
    config = resolve_physical_config("research_hbf_read_write")
    core = NandCore(config, timeline=ResourceTimeline(), capture_details=False)
    now = 0.
    # The same eight-page resident set is physically initialized before every
    # independent ratio row; initialization is outside the measured window.
    for index in range(8):
        now = core.submit(AccessRequest(f"seed-{index}", Operation.WRITE, index*PAGE, PAGE, now)).completion_ns
    start = now
    busy_before, bytes_before = dict(core.timeline.busy_ns), dict(core.timeline.bytes_moved)
    results = []
    for index in range(reads_per_write):
        result = core.submit(AccessRequest(f"read-{index}", Operation.READ, (index % 8)*PAGE, PAGE, now))
        now = result.completion_ns
        results.append(result)
    result = core.submit(AccessRequest("write-0", Operation.WRITE, 0, PAGE, now))
    now = result.completion_ns
    results.append(result)
    duration = now-start
    logical_read = sum(r.logical_bytes for r in results if r.operation is Operation.READ)
    logical_write = sum(r.logical_bytes for r in results if r.operation is Operation.WRITE)
    physical_read = sum(int(r.counters.get("physical_read_bytes", 0)) for r in results)
    physical_write = sum(int(r.counters.get("physical_write_bytes", 0)) for r in results)
    checks = {"exact_read_write_ratio": logical_read == reads_per_write*logical_write,
              "fixed_write_volume": logical_write == PAGE,
              "full_page_media_byte_conservation": physical_read == logical_read and physical_write == logical_write,
              "positive_measured_service": duration > 0}
    return {
        "mode": "physical_nand_read_write_microbenchmark", "runnable": True, "requested_read_write_ratio": reads_per_write,
        "status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "page_bytes": PAGE, "resident_pages": 8, "read_operations": reads_per_write, "write_operations": 1,
        "logical_read_bytes": logical_read, "logical_write_bytes": logical_write,
        "physical_read_bytes": physical_read, "physical_write_bytes": physical_write,
        "observed_read_write_ratio": logical_read/logical_write,
        "initialization_duration_ns": start, "initialization_physical_write_bytes": 8*PAGE,
        "measurement_window": {"start_ns": start, "end_ns": now, "duration_ns": duration},
        "memory_service_duration_ns": duration,
        "effective_logical_read_gb_s": logical_read/duration,
        "effective_logical_write_gb_s": logical_write/duration,
        "effective_total_payload_gb_s": (logical_read+logical_write)/duration,
        "resource_bytes": _delta(dict(core.timeline.bytes_moved), bytes_before),
        "resource_busy_ns": _delta(dict(core.timeline.busy_ns), busy_before),
        "applied_hardware": {"kind": "hbf", "channels": core.config.channels,
            "parallel_units": core.config.parallel_units,
            "host_read_gb_s": core.config.host_read_bandwidth_gb_s,
            "host_write_gb_s": core.config.host_write_bandwidth_gb_s,
            "internal_channel_gb_s": core.config.internal_bandwidth_gb_s,
            "page_read_ns": core.config.page_read_ns, "page_program_ns": core.config.page_program_ns,
            "write_buffer_bytes": core.config.write_buffer_bytes},
        "executed_physical_configurations": {"hbf": {"core": type(core).__name__,
            "physical_config": asdict(core.config)}},
        "node_tps": None, "per_gpu_tps": None, "ttft_ns": None, "tpot_ns": None,
        "comparison_scope": "one full-page write is held fixed; read count and total traffic increase with the ratio. Completion times are not fixed-work speedups. Requests are deliberately sequential.",
    }


def run_traffic_study() -> dict[str, Any]:
    rows = []
    for hits in (50, 80, 95, 99):
        rows.append({"id": f"hot_reads/{hits}", "sweep": "hot_read_fraction", **run_hot_read_trace(hot_reads=hits)})
    for pages in (0, 10, 25, 50):
        rows.append({"id": f"capacity_fraction/{pages}", "sweep": "hbm_capacity_fraction",
                     **run_hot_read_trace(hot_reads=80, hbm_pages=pages)})
    for ratio in (1, 10, 100, 1000):
        rows.append({"id": f"read_write/{ratio}", "sweep": "read_write_ratio", **run_read_write_trace(ratio)})
    executed = [r for r in rows if r["status"] != "configuration_rejected"]
    rejected = [r for r in rows if r["status"] == "configuration_rejected"]
    checks = {
        "all_executed_physical_checks_pass": all(all(row["checks"].values()) for row in executed),
        "only_zero_hbm_staging_is_rejected": len(rejected) == 1 and rejected[0]["hbm_capacity_pages"] == 0,
        "hot_read_targets_are_exact": {row["observed_hot_read_fraction"] for row in rows if row["sweep"] == "hot_read_fraction"} == {.5, .8, .95, .99},
        "capacity_fractions_are_applied": {row["hbm_fraction"] for row in rows if row["sweep"] == "hbm_capacity_fraction"} == {0., .1, .25, .5},
        "read_write_ratios_are_executed": {row["observed_read_write_ratio"] for row in rows if row["sweep"] == "read_write_ratio"} == {1., 10., 100., 1000.},
    }
    passed = all(checks.values())
    return {
        "id": "10_traffic_and_tiering", "title": "⑩ 真实热读命中、容量比例与读写流量",
        "status": "passed" if passed else "failed", "rows": rows, "checks": checks,
        "scope": "production physical memory cores, dynamic KV pool and physical transfers; microbenchmarks only",
        "capability": {"supported": ["显式页序列产生准确50/80/95/99%热读，冷读执行真实双向迁移", "固定总容量的0/10/25/50%HBM预算与staging拒绝", "固定一页写入下实际1/10/100/1000页读取的分向有效带宽"] if passed else [],
                       "unsupported": ["GPU TPS/TTFT/TPOT或生产服务尾延迟", "预测器命中率或最优动态冷热placement", "真实硅片参数和封装校准"]},
        "limitations": [
            "All device parameters and the 100-page capacity budget are explicit synthetic assumptions, not commercial hardware specifications.",
            "The hot set remains eight pages across capacity ratios; extra HBM is not automatically filled. This isolates applied capacity without pretending to optimize placement.",
            "0% HBM is rejected only by the HBM-staging policy. A memory-accessible all-KV-HBF architecture can use a different policy.",
            "Observed hit rates use actual page ownership before each read; they are controlled trace conditions, not learned-predictor accuracy.",
            "Each cold read is followed by an explicit write-back to HBF. This intentionally includes the write cost of the stated policy and is not an optimal immutable-page discard policy.",
            "Initialization is physically executed and separately timed. Measured windows contain every requested read, required restore and return, or the specified read/write mix.",
            "Read/write ratio rows keep one page write fixed while varying read traffic. Use byte-normalized rates, not completion-time ratios, when comparing these different workloads.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_traffic_study()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": len(report["rows"]), "checks": report["checks"]}))
    raise SystemExit(0 if report["status"] == "passed" else 1)
