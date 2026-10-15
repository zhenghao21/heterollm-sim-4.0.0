"""媒体实验组 05--07 的可复现实验入口。

本模块把原始执行计划中的写缓冲、direct/relay 路径和四槽预算单独输出。
它只报告本实验真正测量到的量；内存微基准没有 token 服务语义，因此
``node_tps``/``ttft_ns`` 保持为 ``None``，并附带 ``not_applicable`` 说明。

运行::

    python scripts/hbf_hbm_media_experiments.py --output artifacts/media_experiments.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from heterollm_sim.communication import TopologyRouter
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.ir import ComponentSpec, HardwareSpec, LinkSpec
from heterollm_sim.kv_pool import KvPoolTransferRequest
from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
from heterollm_sim.memory_allocator import AllocationError
from heterollm_sim.memory_transfer import ResourceTimeline, summarize_batch
from heterollm_sim.memory_types import (
    AccessRequest,
    Operation,
)
from heterollm_sim.nand_core import NandCore
from heterollm_sim.physical_presets import resolve_physical_config
from heterollm_sim.physical_contract import reconfigure_physical_config
from hbf_hbm_sessions import _executed_core_configurations

PAGE = 16_384
LARGE_PAGE_ASSUMPTION = 1_572_864
PAGE_SIZE_SCAN = (4 * 1024, PAGE, 64 * 1024, 256 * 1024, LARGE_PAGE_ASSUMPTION)


def _status(checks: Mapping[str, bool]) -> str:
    """Derive status from checks instead of declaring every row successful."""

    values = [bool(value) for value in checks.values()]
    if not values or not any(values):
        return "failed"
    if all(values):
        return "passed"
    return "partial"


def _microbenchmark_availability() -> dict[str, str]:
    return {
        "node_tps": "not_applicable: media microbenchmark has no token-serving scheduler",
        "per_gpu_tps": "not_applicable: media microbenchmark has no GPU serving trace",
        "ttft_ns": "not_applicable: no request prefill event",
        "tpot_ns": "not_applicable: no generated-token event",
    }


def _write_core(*, write_buffer_bytes: int, program_ns: float,
                page_bytes: int = PAGE, total_bytes: int = 8 * PAGE) -> NandCore:
    config = resolve_physical_config("research_hbf_media_write", overrides={
        "capacity_bytes": max(16, (total_bytes + page_bytes - 1) // page_bytes) * page_bytes,
        "page_bytes": page_bytes, "page_program_ns": float(program_ns),
        "write_buffer_bytes": int(write_buffer_bytes)})
    return NandCore(config, timeline=ResourceTimeline(), capture_details=True)


def _append_trace(*, append_bytes: int, write_buffer_bytes: int, program_ns: float,
                  total_bytes: int = 8 * PAGE, page_bytes: int = PAGE) -> dict[str, Any]:
    """Run one fixed-logical-volume append trace and return raw counters."""

    if append_bytes <= 0 or total_bytes <= 0 or page_bytes <= 0:
        raise ValueError("append_bytes, total_bytes and page_bytes must be positive")
    core = _write_core(write_buffer_bytes=write_buffer_bytes, program_ns=program_ns,
                       page_bytes=page_bytes, total_bytes=total_bytes)
    results = []
    address = 0
    remaining = int(total_bytes)
    index = 0
    while remaining:
        size = min(int(append_bytes), remaining)
        results.append(core.submit(AccessRequest(
            f"append-{index}", Operation.WRITE, address, size, 0.0,
        )))
        address += size
        remaining -= size
        index += 1
    batch = summarize_batch(results)
    host_ack_ns = max((float(item.counters.get("host_ack_completion_ns")
                         or item.completion_ns) for item in results), default=0.0)
    drain = core.flush_write_buffer(now_ns=host_ack_ns)
    counters = [item.counters for item in results]
    logical_write = sum(int(item.logical_bytes) for item in results)
    logical_read = sum(int(item.logical_bytes) for item in results
                       if item.operation is Operation.READ)
    link_write = sum(int(item.get("host_transfer_bytes", 0)) for item in counters)
    link_read = 0
    media_write = sum(int(item.get("physical_write_bytes", 0)) for item in counters) + drain.get("physical_write_bytes", 0)
    media_read = sum(int(item.get("physical_read_bytes", 0)) for item in counters) + drain.get("physical_read_bytes", 0)
    merge_bytes = sum(int(item.get("merge_bytes", 0)) for item in counters)
    blocking_ns = sum(float(item.get("write_buffer_wait_ns", 0.0)) for item in counters)
    peak = max((int(item.get("write_buffer_peak_bytes", 0)) for item in counters), default=0)
    media_end = float(drain.get("media_drain_completion_ns", host_ack_ns))
    measured = {
        "measurement_window_ns": media_end,
        "measurement_window": "arrival at zero through final media drain, including RMW, queueing and buffer flush",
        "effective_logical_read_bandwidth_gb_s": None,
        "effective_logical_write_bandwidth_gb_s": logical_write / media_end if media_end else None,
        "host_ack_write_bandwidth_gb_s": logical_write / host_ack_ns if host_ack_ns else None,
        "media_read_bandwidth_gb_s": media_read / media_end if media_end else None,
        "media_write_bandwidth_gb_s": media_write / media_end if media_end else None,
        "write_amplification": media_write / logical_write if logical_write else None,
        "page_write_payload_utilization": logical_write / media_write if media_write else None,
        "host_write_peak_fraction": logical_write / media_end / 16.0 if media_end else None,
        "channel_utilization": {f"nand:channel:{i}": core.timeline.busy_ns.get(f"nand:channel:{i}", 0.0) / media_end
                                for i in range(core.config.channels)} if media_end else {},
        "parallel_unit_utilization": {f"nand:parallel:{i}": core.timeline.busy_ns.get(f"nand:parallel:{i}", 0.0) / media_end
                                      for i in range(4)} if media_end else {},
        "scope": "observed base append trace, before the analytical GC overlay; page utilization is logical append bytes / programmed page bytes",
        "unavailable": {"effective_logical_read_bandwidth_gb_s": "no logical reads in this write-only trace; physical reads are RMW",
                        "plane_imbalance": "four abstract service lanes do not establish physical NAND plane geometry",
                        "read_program_queue_depth": "timeline measures wait duration, not pending request queue depth"},
    }
    return {
        "append_bytes": int(append_bytes),
        "page_bytes": int(page_bytes),
        "applied_physical_page_bytes": int(core.config.page_bytes),
        "page_size_semantics": "NandCore physical read/program page; independent of group 06 migration payload",
        "page_size_assumption": (
            "requested 1.5MiB large-page sensitivity assumption; not a vendor NAND page specification"
            if page_bytes == LARGE_PAGE_ASSUMPTION else
            "uncalibrated physical-page sensitivity assumption"
        ),
        "total_logical_append_bytes": int(total_bytes),
        "write_buffer_capacity_bytes": int(write_buffer_bytes),
        "program_ns": float(program_ns),
        "append_count": len(results),
        "logical_read_bytes": logical_read,
        "logical_write_bytes": logical_write,
        "link_read_bytes": link_read,
        "link_write_bytes": link_write,
        "media_read_bytes": media_read,
        "media_write_bytes": media_write,
        "host_ack_completion_ns": host_ack_ns,
        "media_drain_start_ns": float(drain.get("media_drain_start_ns", host_ack_ns)),
        "media_drain_completion_ns": float(drain.get("media_drain_completion_ns", host_ack_ns)),
        "host_ack_to_media_drain_ns": max(
            0.0,
            float(drain.get("media_drain_completion_ns", host_ack_ns)) - host_ack_ns,
        ),
        "merge_bytes": merge_bytes,
        "flushed_pages": int(drain.get("flushed_pages", 0)),
        "physical_programs": sum(item.pages_programmed for item in results) + int(drain.get("physical_programs", 0)),
        "write_buffer_peak_bytes": peak,
        "write_buffer_remaining_bytes_after_drain": core.pending_write_buffer_bytes,
        "buffered_pages_remaining_after_drain": core.buffered_page_count,
        "write_buffer_blocking_ns": blocking_ns,
        "queue_wait_ns": float(batch.counters.get("queue_wait_ns", 0.0)),
        "resource_bytes": dict(core.timeline.bytes_moved),
        "resource_busy_ns": dict(core.timeline.busy_ns),
        "observed_metrics": measured,
        "node_tps": None,
        "ttft_ns": None,
        "availability": _microbenchmark_availability(),
    }


def run_group_05_write_buffer() -> dict[str, Any]:
    """Run the fixed append-volume write-buffer scan."""

    rows = [
        _append_trace(append_bytes=append, write_buffer_bytes=capacity, program_ns=program,
                      page_bytes=page)
        for page in PAGE_SIZE_SCAN
        for append in (4 * 1024, PAGE // 2, PAGE)
        for capacity in (0, PAGE, 2 * PAGE, 4 * PAGE)
        for program in (75_000.0, 100_000.0, 200_000.0)
    ]
    expanded = []
    for row in rows:
        for amplification in (1, 3, 10, 30):
            extra_bytes = row["media_write_bytes"] * (amplification - 1)
            # Each hypothetical GC relocation reads then programs one page.
            # The base trace already includes its own RMW and drain work;
            # only WA - 1 relocation work is added. No host bytes are added.
            extra_read_service = extra_bytes / row["page_bytes"] * 4_000.0 / 4
            extra_program_service = extra_bytes / row["page_bytes"] * row["program_ns"] / 4
            extra_transfer_service = 2 * extra_bytes / 128.0 / 4
            extra_service = extra_read_service + extra_program_service + extra_transfer_service
            expanded.append({**row, "gc_write_amplification": amplification,
                "base_media_read_bytes": row["media_read_bytes"],
                "base_media_write_bytes": row["media_write_bytes"],
                "media_read_bytes": row["media_read_bytes"] + extra_bytes,
                "media_write_bytes": row["media_write_bytes"] * amplification,
                "gc_extra_media_read_bytes": extra_bytes,
                "gc_extra_media_write_bytes": extra_bytes,
                "gc_overlay_extra_read_ns": extra_read_service,
                "gc_overlay_extra_program_ns": extra_program_service,
                "gc_overlay_extra_internal_transfer_ns": extra_transfer_service,
                "gc_overlay_extra_media_service_ns": extra_service,
                "gc_overlay_drain_completion_ns": row["media_drain_completion_ns"] + extra_service,
                "gc_assumed_write_amplification": row["media_write_bytes"] * amplification / row["logical_write_bytes"],
                "gc_assumed_effective_write_bandwidth_gb_s": row["logical_write_bytes"] / (row["media_drain_completion_ns"] + extra_service),
                "gc_model": "analytical page-relocation read/program overlay evenly spread across four parallel units; no FTL state"})
    rows = expanded
    total = 8 * PAGE
    checks = {
        "fixed_logical_append_volume": all(
            row["logical_write_bytes"] == total for row in rows
        ),
        "finite_buffer_peak_is_bounded": all(
            row["write_buffer_peak_bytes"] <= row["write_buffer_capacity_bytes"]
            for row in rows
        ),
        "page_merge_is_observable": any(row["merge_bytes"] > 0 for row in rows),
        "buffer_full_causes_blocking": any(
            row["write_buffer_blocking_ns"] > 0 for row in rows
        ),
        "host_ack_precedes_media_drain": any(
            row["host_ack_to_media_drain_ns"] > 0 for row in rows
        ),
        "program_time_changes_drain": any(
            slow["media_drain_completion_ns"] > fast["media_drain_completion_ns"]
            for fast in rows if fast["program_ns"] == 75_000.0
            for slow in rows
            if slow["program_ns"] == 200_000.0
            and slow["page_bytes"] == fast["page_bytes"]
            and slow["append_bytes"] == fast["append_bytes"]
            and slow["write_buffer_capacity_bytes"] == fast["write_buffer_capacity_bytes"]
        ),
        "gc_overlay_does_not_multiply_host_traffic": all(row["link_write_bytes"] == total for row in rows),
        "all_gc_factors_present": {row["gc_write_amplification"] for row in rows} == {1, 3, 10, 30},
        "drained_program_byte_conservation": all(row["base_media_write_bytes"] == row["physical_programs"] * row["page_bytes"] for row in rows),
        "physical_page_sizes_present": {row["page_bytes"] for row in rows} == set(PAGE_SIZE_SCAN),
        "physical_page_sizes_applied_to_core": all(row["applied_physical_page_bytes"] == row["page_bytes"] for row in rows),
        "all_buffers_fully_drained": all(row["write_buffer_remaining_bytes_after_drain"] == 0
                                         and row["buffered_pages_remaining_after_drain"] == 0 for row in rows),
        "large_page_partial_writes_include_rmw": all(
            row["base_media_read_bytes"] == row["base_media_write_bytes"]
            and row["base_media_write_bytes"] >= LARGE_PAGE_ASSUMPTION
            for row in rows if row["page_bytes"] == LARGE_PAGE_ASSUMPTION),
        "observed_metrics_keep_gc_overlay_separate": all(row["observed_metrics"]["write_amplification"] == row["base_media_write_bytes"] / total for row in rows),
        "gc_overlay_adds_only_extra_relocation_work": all(
            row["gc_extra_media_read_bytes"] == row["gc_extra_media_write_bytes"]
            == row["base_media_write_bytes"] * (row["gc_write_amplification"] - 1)
            for row in rows),
    }
    return {
        "id": "05_write_buffer",
        "status": _status(checks),
        "scope": "固定128KiB逻辑append；扫描4/16/64/256KiB及1.5MiB假设物理页、append大小、有限缓冲和program时间",
        "rows": rows,
        "checks": checks,
        "capability": {
            "supported": [
                "固定逻辑写入量下的页合并、host ack/media drain 分离和满缓冲阻塞",
                "逻辑流量、host link 流量与介质 page program 流量的分离统计",
            ] if _status(checks) == "passed" else [],
            "unsupported": [
                "GC/FTL 状态机和真实 GC 写放大长尾未实现；1/3/10/30 倍只作额外介质页搬运读写分析叠加",
                "TPS、TTFT 不适用于该介质微基准",
            ],
        },
        "limitations": [
            "使用生产 NandCore 的简化页合并/缓冲模型，不推断真实控制器 GC 长尾。",
            "rows 保留所有扫描组合，不把最快或最佳行当作组结论。",
            "页大小变化时逻辑写量、append大小与缓冲字节数保持不变；缓冲为0/16/32/64KiB，容量按host payload而非整页计费。",
            "observed_metrics只来自实际append与排空；gc_assumed字段为分析假设，不冒充已执行GC。",
            "1.5MiB在本组作为附件要求的大页假设，实际设置NandCore物理read/program页大小；这不是厂商NAND页规格或已校准硬件声明。",
            "1.5MiB页大于固定128KiB逻辑工作量，因此所有写入都是部分页写入；计入完整大页的读改写成本，不能把它解读为吞吐率优势。",
            "05组的物理页大小与06组的迁移payload独立；06组即使迁移块为1.5MiB，其NAND物理页仍是16KiB。",
        ],
    }


def _migration_hardware(*, relay_capacity_bytes: int | None = None,
                        minimum_capacity_bytes: int = 0) -> HardwareSpec:
    capacity = max(128 * PAGE, ((minimum_capacity_bytes + PAGE - 1) // PAGE) * PAGE)
    hbm = resolve_physical_config("research_hbm_relay", overrides={"capacity_bytes": capacity})
    hbf = resolve_physical_config("research_hbf_migration", overrides={"capacity_bytes": capacity})
    components = (
        ComponentSpec("hbf", "hbf", capacity_bytes=capacity,
                      read_bandwidth_gbps=1024.0, write_bandwidth_gbps=256.0,
                      metadata={"physical_memory_config": hbf}),
        ComponentSpec("gpu_hbm", "hbm", capacity_bytes=capacity,
                      bandwidth_gbps=2048.0, metadata={"physical_memory_config": hbm}),
        ComponentSpec("relay_hbm", "hbm", capacity_bytes=relay_capacity_bytes or capacity,
                      bandwidth_gbps=2048.0, metadata={"physical_memory_config": reconfigure_physical_config(
                          hbm, overrides={"capacity_bytes": relay_capacity_bytes or capacity})}),
    )
    links = (
        LinkSpec("direct", "hbf", "d", "gpu_hbm", "d", "pcie",
                 bandwidth_gbps=32.0, latency_ns=50.0),
        LinkSpec("relay_in", "hbf", "r", "relay_hbm", "in", "pcie",
                 bandwidth_gbps=64.0, latency_ns=50.0),
        LinkSpec("relay_out", "relay_hbm", "out", "gpu_hbm", "r", "pcie",
                 bandwidth_gbps=64.0, latency_ns=50.0),
    )
    return HardwareSpec("media-path", components, links, require_connected=False)


def _run_path_row(path: str, payload_bytes: int = 16 * PAGE, *, relay_capacity_bytes: int | None = None) -> dict[str, Any]:
    hardware = _migration_hardware(relay_capacity_bytes=relay_capacity_bytes,
                                   minimum_capacity_bytes=2 * payload_bytes)
    runtime = PhysicalRuntimeContext(capture_details=False)
    direct_topology = TopologyRouter(replace(hardware, links=hardware.links[:1]))
    relay_topology = TopologyRouter(replace(hardware, links=hardware.links[1:]))
    flows = []
    # Both flows use the same physical source and target, so their medium and
    # host resources remain shared even when independent fabric paths overlap.
    for flow in ("weight", "kv"):
        use_relay = path == "relay" or (path == "dual" and flow == "weight")
        executor = PhysicalKVTransferExecutor(
            hardware.components, runtime=runtime,
            topology=relay_topology if use_relay else direct_topology,
            relay_components=("relay_hbm",) if use_relay else (), namespace=flow,
        )
        page = SimpleNamespace(logical_page_id=0, request_id=flow, layer_group=flow,
                               owner_component="hbf", bytes=payload_bytes, allocation_generation=0)
        executor.bind_page(page)
        # Keep both source allocations live before submitting either transfer.
        flows.append((flow, executor, page, use_relay))
    rows = []
    for flow, executor, page, use_relay in flows:
        plan = executor(KvPoolTransferRequest(
            page.logical_page_id, flow, flow, "hbf", "gpu_hbm", payload_bytes, 0, 0, 1))
        plan.commit()
        event = executor.events[-1]
        phases = list(event["phases"])
        rows.append({"flow": flow, "route": "relay" if use_relay else "direct",
                     "logical_read_bytes": payload_bytes, "logical_write_bytes": payload_bytes,
                     "link_bytes": executor.counters["link_bytes"],
                     "relay_bytes": executor.counters["relay_bytes"],
                     "completion_ns": plan.completion_ns,
                     "physical_read_bytes": sum(p.get("physical_read_bytes", 0) for p in phases),
                     "physical_write_bytes": sum(p.get("physical_write_bytes", 0) for p in phases),
                     "source_hbf_physical_read_bytes": sum(p.get("physical_read_bytes", 0) for p in phases if p["kind"] == "source_read"),
                     "source_resource_ids": sorted({resource for p in phases if p["kind"] == "source_read" for resource in p.get("resource_busy_ns", {})}),
                     "target_resource_ids": sorted({resource for p in phases if p["kind"] == "target_write" for resource in p.get("resource_busy_ns", {})}),
                     "link_phases": [p for p in phases if p["kind"] == "link"]})
    makespan = max(row["completion_ns"] for row in rows)
    return {"path": path, "payload_bytes_per_flow": payload_bytes,
            "logical_read_bytes": 2 * payload_bytes, "logical_write_bytes": 2 * payload_bytes,
            "flows": rows, "completion_ns": makespan,
            "useful_bandwidth_gb_s": 2 * payload_bytes / makespan,
            "link_bytes": sum(row["link_bytes"] for row in rows),
            "relay_bytes": sum(row["relay_bytes"] for row in rows),
            "physical_read_bytes": sum(row["physical_read_bytes"] for row in rows),
            "physical_write_bytes": sum(row["physical_write_bytes"] for row in rows),
            "resource_bytes": dict(runtime.timeline.bytes_moved),
            "resource_busy_ns": dict(runtime.timeline.busy_ns),
            "executed_physical_configurations": _executed_core_configurations(runtime),
            "observed_metrics": {
                "measurement_window_ns": makespan,
                "measurement_window": "both flows released at zero through last target-memory completion",
                "effective_hbf_read_payload_gb_s": 2 * payload_bytes / makespan,
                "effective_hbm_write_payload_gb_s": 2 * payload_bytes / makespan,
                "effective_hbf_write_payload_gb_s": None,
                "hbf_read_logical_to_media_byte_ratio": 2 * payload_bytes / sum(row["source_hbf_physical_read_bytes"] for row in rows)
                    if sum(row["source_hbf_physical_read_bytes"] for row in rows) else None,
                "scope": "end-to-end directional payload rates, including media and path delays; relay bytes excluded from useful payload",
                "link_utilization": {key: busy / makespan for key, busy in runtime.timeline.busy_ns.items()
                                     if key.startswith("link.")},
                "unavailable": {"effective_hbf_write_payload_gb_s": "this migration trace only reads the HBF source"},
            },
            "configured_capacity_bytes": {component.component_id: component.capacity_bytes for component in hardware.components},
            "final_allocated_bytes": {owner: sum(item.size_bytes for item in allocator.allocations()) for owner, allocator in runtime.allocators.items()},
            "node_tps": None, "ttft_ns": None, "availability": _microbenchmark_availability()}


def run_group_06_path_direct_relay() -> dict[str, Any]:
    rows = [_run_path_row(path) for path in ("direct", "relay", "dual")]
    block_size_scan = [_run_path_row(path, payload_bytes=block)
                       for block in (4096, 16384, 65536, 262144, 1572864)
                       for path in ("direct", "relay", "dual")]
    by_path = {row["path"]: row for row in rows}
    payload = rows[0]["payload_bytes_per_flow"]
    dual_links = [phase for flow in by_path["dual"]["flows"] for phase in flow["link_phases"]]
    overlap = any(a["resource_id"] != b["resource_id"] and
                  max(a["start_ns"], b["start_ns"]) < min(a["completion_ns"], b["completion_ns"])
                  for a in dual_links for b in dual_links)
    staging_capacity_enforced = False
    try:
        _run_path_row("relay", relay_capacity_bytes=PAGE)
    except AllocationError:
        staging_capacity_enforced = True
    checks = {
        "same_payload_and_endpoints": all(row["logical_read_bytes"] == 2 * payload and
                                           row["logical_write_bytes"] == 2 * payload for row in rows),
        "actual_direct_relay_dual_paths": all(
            [flow["route"] for flow in by_path[path]["flows"]] == routes
            for path, routes in (("direct", ["direct", "direct"]),
                                 ("relay", ["relay", "relay"]), ("dual", ["relay", "direct"]))),
        "link_byte_conservation": [row["link_bytes"] for row in rows] == [2 * payload, 4 * payload, 3 * payload],
        "relay_memory_is_charged": [row["relay_bytes"] for row in rows] == [0, 4 * payload, 2 * payload],
        "physical_media_byte_conservation": all(row["physical_read_bytes"] == row["logical_read_bytes"] + row["relay_bytes"] // 2
                                                and row["physical_write_bytes"] == row["logical_write_bytes"] + row["relay_bytes"] // 2
                                                for row in rows),
        "dual_path_has_real_fabric_overlap": overlap,
        "relay_staging_capacity_is_enforced": staging_capacity_enforced,
        "migration_block_sizes_present": {row["payload_bytes_per_flow"] for row in block_size_scan} == {4096, 16384, 65536, 262144, 1572864},
        "flows_share_endpoint_resource_queues": all(
            any(resource.endswith(":nand:host") for resource in
                set(row["flows"][0]["source_resource_ids"]) & set(row["flows"][1]["source_resource_ids"]))
            and any(":dram:data:" in resource for resource in
                set(row["flows"][0]["target_resource_ids"]) & set(row["flows"][1]["target_resource_ids"]))
            for row in rows),
    }
    return {"id": "06_path_direct_relay", "status": _status(checks), "checks": checks, "rows": rows,
            "block_size_scan": block_size_scan,
            "block_size_semantics": "migration payload per flow; NAND physical page remains 16KiB, including the 1.5MiB block case",
            "scope": "同一 HBF→GPU HBM 的权重与 KV 等量混流；direct、HBM relay、双路径共享介质队列",
            "capability": {"supported": ["两条真实链路路径并发、relay 读写计费、源目标介质共享和字节守恒"] if _status(checks) == "passed" else [],
                           "unsupported": ["包级协议、流间抢占、公平仲裁和 token 服务调度未建模"]},
            "limitations": ["权重和 KV 均为固定 payload 的物理传输微基准；结果不是 TPS。",
                            "双路径使用原硬件图中已有的 direct 与 relay 链路，没有凭空增加介质带宽。",
                            "基准按权重、KV 顺序预约事件，采用整块传输，不代表最优包级调度。"]}


def run_group_07_package_budget() -> dict[str, Any]:
    """Run all normalized four-slot mixes, retaining toy checks separately."""

    budget = importlib.import_module("hbf_hbm_budget")
    report = budget.run_architecture_budget_sweep()
    accounting = budget.run_budget_sweep()
    rows = list(report.get("rows", ()))
    runnable = [row for row in rows if row.get("runnable")]
    checks = {**report["checks"],
        "four_slot_budget_is_preserved": report.get("total_slots") == 4,
        "capacity_failure_has_reason": any(
            row.get("runnable") is False and bool(row.get("reason")) for row in rows
        ),
        "runnable_rows_keep_fixed_logical_traffic": (
            bool(runnable)
            and len({row.get("logical_bytes") for row in runnable}) == 1
        ),
        "rows_are_raw_budget_points": len(rows) == 5 and all("hbm_slots" in row for row in rows),
        "eightfold_hbf_density": report["slot_profiles"]["hbf"]["capacity_pages"] == 8 * report["slot_profiles"]["hbm"]["capacity_pages"],
    }
    status = _status(checks)
    return {
        "id": "07_package_budget",
        "title": "⑦ 固定四槽封装预算",
        "status": status,
        "scope": "归一化四槽封装，HBF单槽容量为HBM的8倍；固定40页工作集，联动容量、方向带宽、并行度与缓冲",
        "rows": rows,
        "checks": checks,
        "slot_profiles": report["slot_profiles"],
        "accounting_checks": {
            "scope": "legacy regression only; deliberately inverted capacities do not represent an architecture comparison",
            "report": accounting,
        },
        "capability": {
            "supported": [
                "HBF保持8倍归一化单槽容量时，全部五种HBM/HBF配比的可运行性与实际内存服务代价",
                "容量不足行及不可运行原因",
            ] if status == "passed" else [],
            "unsupported": [
                "真实封装、面积和商业产品校准",
                "商用最优比例、完整GPU吞吐、成本或热约束下的配置选择",
                "TPS、TTFT 不适用于预算微基准",
            ],
        },
        "limitations": report["limitations"],
    }


def run_media_experiments() -> dict[str, Any]:
    groups = [run_group_05_write_buffer(), run_group_06_path_direct_relay(), run_group_07_package_budget()]
    return {
        "schema_version": 1,
        "scope": "原始执行计划的媒体实验组 05--07；不是完整论文复现",
        "groups": groups,
        "status": _status({group["id"]: group["status"] == "passed" for group in groups}),
        "capability": {
            "supported": [capability for group in groups for capability in group["capability"]["supported"]],
            "unsupported": [limitation for group in groups for limitation in group["capability"]["unsupported"]],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(run_media_experiments(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
