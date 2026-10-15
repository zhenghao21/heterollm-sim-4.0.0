"""Repeatable single-device HBM/HBF physical-memory acceptance experiments.

Run ``python scripts/accept_hbf_hbm.py --output artifacts/hbf_hbm.json``.
No installation or third-party dependencies are required. Results are memory
microbenchmarks, not LLM token throughput or calibrated silicon predictions.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.dram_core import DramCore
from heterollm_sim.communication import TopologyRouter
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.ir import (
    ComponentSpec,
    HardwareSpec,
    KVCachePolicy,
    LinkSpec,
    PortSpec,
    RequestSpec,
)
from heterollm_sim.kv_pool import DynamicKVPool, KvPoolComponent
from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
from heterollm_sim.memory_transfer import ResourceTimeline, summarize_batch
from heterollm_sim.memory_types import (
    AccessRequest, DramConfig, MemoryKind, NandConfig, Operation,
    make_hbf_config, make_hbm_config, make_ssd_config,
)
from heterollm_sim.nand_core import NandCore
from heterollm_sim.physical_presets import resolve_physical_config
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serving import BatchCost, _OnlineRuntime, compile_serving_plan

PAGE = 16_384
CAPACITY = 32 * 1024 * 1024


def _serving_paged_scenario(
    *, hbf_capacity: int, requests=(), hbm_capacity: int = 32 * PAGE,
    prefetch_distance: int = 0, max_num_seqs: int = 1
):
    """Build a tiny physical HBM+HBF serving scenario for acceptance."""

    base = build_reference_scenario()
    hbm = base.hardware.get_component("hbm0")
    hbm = replace(
        hbm,
        capacity_bytes=hbm_capacity,
        metadata={
            **hbm.metadata,
            "physical_memory_config": resolve_physical_config(
                "research_hbm_sessions", overrides={"capacity_bytes": hbm_capacity}),
        },
    )
    hbf_physical = resolve_physical_config(
        "research_hbf_sessions", overrides={"capacity_bytes": hbf_capacity})
    hbf = ComponentSpec(
        component_id="hbf0",
        kind="hbf",
        ports=(
            PortSpec(
                port_id="pcie0",
                protocol="PCIe",
                role="endpoint",
                version="5.0",
                lanes=16,
                bandwidth_gbps=512.0,
                payload="coherent_dma",
            ),
        ),
        capacity_bytes=hbf_capacity,
        read_bandwidth_gbps=488.0 * 8.0,
        write_bandwidth_gbps=27.2 * 8.0,
        metadata={
            "physical_memory_config": hbf_physical,
            "access_mode": "remote_flash",
        },
    )
    hardware = replace(
        base.hardware,
        components=tuple(
            hbm if component.component_id == "hbm0" else component
            for component in base.hardware.components
        ) + (hbf,),
        links=base.hardware.links
        + (
            LinkSpec(
                link_id="cpu-hbf-pcie",
                source_component="cpu0",
                source_port="pcie0",
                target_component="hbf0",
                target_port="pcie0",
                protocol="PCIe",
                version="5.0",
                lanes=16,
                bandwidth_gbps=512.0,
                latency_ns=100.0,
                payload="coherent_dma",
            ),
        ),
        require_connected=False,
    )
    policy = KVCachePolicy(
        cache_component="hbm0",
        offload_component="hbf0",
        tokens_per_page=16,
        dtype="int8",
        layout_mode="paged_pool",
        pool_components=("hbm0", "hbf0"),
        prefetch_distance=prefetch_distance,
        preemption_mode="swap",
    )
    scheduler = replace(
        base.workload.scheduler,
        max_num_seqs=max(1, int(max_num_seqs)),
        max_num_batched_tokens=32,
        max_num_ubatch_tokens=32,
        prefill_chunk_tokens=32,
        preemption_enabled=True,
        preemption_policy="swap",
    )
    workload = replace(
        base.workload,
        requests=tuple(requests),
        request_count=len(requests),
        scheduler=scheduler,
        mtp=None,
    )
    return replace(
        base,
        hardware=hardware,
        placement=replace(
            base.placement,
            hardware_name=hardware.name,
            kv_policy=policy,
        ),
        workload=workload,
    )


def check_serving_paged_swap() -> dict[str, Any]:
    """Verify automatic multi-page swap/resume and atomic capacity rollback."""

    requests = (
        RequestSpec(
            request_id="swap-low",
            arrival_ns=0.0,
            prompt_tokens=32,
            output_tokens=8,
            priority=0,
        ),
        RequestSpec(
            request_id="swap-high",
            arrival_ns=1.0,
            prompt_tokens=32,
            output_tokens=8,
            priority=10,
        ),
    )
    plan = compile_serving_plan(
        _serving_paged_scenario(
            hbf_capacity=16 * 1024 * 1024,
            requests=requests,
        )
    )
    runtime = _OnlineRuntime(
        plan,
        lambda *_: BatchCost(100.0),
    )
    result = runtime.run()
    states = runtime.states
    low = states["swap-low"]
    high = states["swap-high"]
    swap_out = [event for event in runtime.events if event.event_type == "kv_swap_out"]
    swap_in = [event for event in runtime.events if event.event_type == "kv_swap_in"]
    event_times = [float(event.timestamp_ns) for event in runtime.events]
    require(event_times == sorted(event_times),
            "serving events must be returned in non-decreasing timestamp order")
    low_events = [
        event for event in runtime.events if event.request_id == "swap-low"
    ]
    low_event_by_type = {}
    for event in low_events:
        low_event_by_type.setdefault(event.event_type, []).append(event)
    low_out_start = low_event_by_type.get("kv_swap_out_start", [None])[0]
    low_out_end = low_event_by_type.get("kv_swap_out", [None])[-1]
    transfer_events = [
        event for event in runtime.events
        if event.event_type in {
            "kv_pool_transfer_start", "kv_pool_transfer_complete"
        }
        and low_out_start is not None
        and low_out_end is not None
        and low_out_start.timestamp_ns <= event.timestamp_ns <= low_out_end.timestamp_ns
    ]
    for event in transfer_events:
        low_event_by_type.setdefault(event.event_type, []).append(event)
    required_low_types = (
        "kv_swap_out_start",
        "kv_pool_transfer_start",
        "kv_pool_transfer_complete",
        "kv_swap_out",
    )
    require(all(low_event_by_type.get(kind) for kind in required_low_types),
            "paged swap must expose ordered physical transfer boundaries")
    low_order = [
        low_event_by_type[kind][0].timestamp_ns
        for kind in required_low_types[:3]
    ] + [low_event_by_type["kv_swap_out"][-1].timestamp_ns]
    require(low_order == sorted(low_order),
            "swap start, page transfer, completion, and swap end are out of order")
    low_in_events = [
        event for event in runtime.events if event.request_id == "swap-low"
        and event.event_type in {"kv_swap_in_start", "kv_swap_in"}
    ]
    require(
        any(event.event_type == "kv_swap_in_start" for event in low_in_events)
        and any(event.event_type == "kv_swap_in" for event in low_in_events),
        "paged restore must expose swap-in boundaries",
    )
    low_in_start = next(
        event for event in low_in_events if event.event_type == "kv_swap_in_start"
    )
    low_in_end = next(
        event for event in reversed(low_in_events) if event.event_type == "kv_swap_in"
    )
    restore_transfer_events = [
        event for event in runtime.events
        if event.event_type in {
            "kv_pool_transfer_start", "kv_pool_transfer_complete"
        }
        and low_in_start.timestamp_ns <= event.timestamp_ns <= low_in_end.timestamp_ns
    ]
    require(
        restore_transfer_events
        and restore_transfer_events[0].event_type == "kv_pool_transfer_start"
        and restore_transfer_events[-1].event_type == "kv_pool_transfer_complete",
        "swap-in page transfers must be enclosed by restore boundaries",
    )
    require(all(state.status.value == "finished" for state in states.values()),
            "automatic paged serving swap must finish every request")
    require(runtime.preemptions >= 1 and runtime.swap_events >= 1,
            "higher-priority arrival must trigger a serving preemption and swap")
    require(swap_out and swap_in and swap_out[0].details["pages"] == 2,
            "automatic swap must transfer the complete multi-page KV working set")
    require(runtime.swap_transfer_time_ns > 0.0,
            "serving swap time must come from physical page transfer events")
    require(low.swaps == 1 and high.swaps == 0,
            "only the lower-priority request should be swapped")
    require(not runtime.ledger.paged_pool.pages() and
            all(value == 0 for value in runtime.physical_ledger.used_bytes.values()),
            "finished requests must release resident and offloaded pages")

    rollback_plan = compile_serving_plan(
        _serving_paged_scenario(
            hbf_capacity=PAGE,
            requests=(
                RequestSpec(
                    request_id="rollback",
                    arrival_ns=0.0,
                    prompt_tokens=32,
                    output_tokens=8,
                ),
            ),
        )
    )
    rollback_runtime = _OnlineRuntime(
        rollback_plan,
        lambda *_: BatchCost(0.0),
    )
    rollback_request = rollback_runtime.states["rollback"]
    pool = rollback_runtime.ledger.paged_pool
    require(pool is not None and rollback_runtime.ledger.resize(rollback_request, 2),
            "rollback setup must allocate two resident pages")
    before_owners = tuple(page.owner_component for page in pool.request_pages("rollback"))
    require(not rollback_runtime.ledger.offload(rollback_request),
            "offload must fail when the HBF target cannot fit all pages")
    after_owners = tuple(page.owner_component for page in pool.request_pages("rollback"))
    require(after_owners == before_owners and rollback_request.kv_pages == 2 and
            rollback_request.swapped_pages == 0,
            "failed multi-page offload must roll back every page and ledger field")
    require(rollback_runtime.physical_ledger.used_bytes.get("hbm0") == 2 * PAGE and
            rollback_runtime.physical_ledger.used_bytes.get("hbf0") == 0,
            "failed offload must not leak target capacity or source accounting")

    restore_plan = compile_serving_plan(
        _serving_paged_scenario(
            hbf_capacity=2 * PAGE,
            hbm_capacity=2 * PAGE,
            requests=(
                RequestSpec(
                    request_id="restore-victim",
                    arrival_ns=0.0,
                    prompt_tokens=32,
                    output_tokens=8,
                ),
                RequestSpec(
                    request_id="restore-owner",
                    arrival_ns=0.0,
                    prompt_tokens=32,
                    output_tokens=8,
                ),
            ),
        )
    )
    restore_runtime = _OnlineRuntime(
        restore_plan,
        lambda *_: BatchCost(1.0),
    )
    victim = restore_runtime.states["restore-victim"]
    owner = restore_runtime.states["restore-owner"]
    restore_pool = restore_runtime.ledger.paged_pool
    require(
        restore_pool is not None
        and restore_runtime.ledger.resize(victim, 2)
        and restore_runtime.ledger.offload(victim)
        and restore_runtime.ledger.resize(owner, 2),
        "restore rollback setup must occupy HBF and all HBM pages",
    )
    restore_before = tuple(
        page.owner_component
        for page in restore_pool.request_pages("restore-victim")
    )
    require(
        not restore_runtime.ledger.restore(victim),
        "restore must fail when another request fills the HBM target",
    )
    restore_after = tuple(
        page.owner_component
        for page in restore_pool.request_pages("restore-victim")
    )
    require(
        restore_after == restore_before
        and victim.kv_pages == 0
        and victim.swapped_pages == 2
        and restore_runtime.physical_ledger.used_bytes.get("hbm0") == 2 * PAGE
        and restore_runtime.physical_ledger.used_bytes.get("hbf0") == 2 * PAGE,
        "failed restore must preserve offloaded pages and both capacity ledgers",
    )
    return {
        "preemptions": runtime.preemptions,
        "swap_events": runtime.swap_events,
        "swap_out_pages": swap_out[0].details["pages"],
        "swap_transfer_time_ns": runtime.swap_transfer_time_ns,
        "makespan_ns": result.makespan_ns,
        "swap_low_ttft_ns": result.request_metrics["swap-low"].ttft_ns,
        "swap_low_tpot_ns": result.request_metrics["swap-low"].tpot_ns,
        "kv_peak_pages": result.kv_metrics.peak_used_pages,
        "physical_swap_bytes": result.kv_metrics.physical_swap_bytes,
        "requests_finished": len(result.request_states),
        "rollback_preserved_pages": len(after_owners),
        "rollback_target_used_bytes": rollback_runtime.physical_ledger.used_bytes.get("hbf0"),
        "restore_rollback_preserved_pages": len(restore_after),
        "restore_target_used_bytes": restore_runtime.physical_ledger.used_bytes.get("hbm0"),
    }


def check_prefetch_distance() -> dict[str, Any]:
    """Verify fixed token lookahead, page rounding, and zero-distance behavior."""

    requests = (
        RequestSpec(
            request_id="prefetch-low",
            arrival_ns=0.0,
            prompt_tokens=32,
            output_tokens=8,
            priority=0,
        ),
        RequestSpec(
            request_id="prefetch-high",
            arrival_ns=1.0,
            prompt_tokens=32,
            output_tokens=8,
            priority=10,
        ),
    )

    def run(distance: int):
        scenario = _serving_paged_scenario(
            hbf_capacity=2 * PAGE,
            requests=requests,
            prefetch_distance=distance,
        )
        plan = compile_serving_plan(scenario)
        runtime = _OnlineRuntime(
            plan,
            lambda *_: BatchCost(1.0),
        )
        return plan, runtime, runtime.run()

    plan, runtime, result = run(17)
    metrics = result.kv_metrics
    prefetch_events = [
        event for event in result.events if event.event_type == "kv_prefetch"
    ]
    require(plan.kv_policy.prefetch_distance == 17,
            "positive prefetch_distance must survive serving compilation")
    require(metrics.prefetch_distance_modeled,
            "positive prefetch_distance must be marked modeled")
    require(metrics.prefetch_events >= 1 and metrics.prefetch_bytes == 2 * PAGE,
            "lookahead must prefetch two rounded-up pages")
    require(prefetch_events,
            "positive prefetch_distance must emit kv_prefetch")
    details = prefetch_events[0].details
    require(details.get("distance_tokens") == 17 and
            details.get("lookahead_pages") == 2 and
            details.get("bytes") == 2 * PAGE,
            "kv_prefetch must expose token distance, page rounding, and bytes")
    require(metrics.prefetch_transfer_time_ns > 0.0,
            "prefetch must consume physical transfer time")

    zero_plan, zero_runtime, zero_result = run(0)
    require(zero_plan.kv_policy.prefetch_distance == 0 and
            not zero_result.kv_metrics.prefetch_distance_modeled and
            zero_result.kv_metrics.prefetch_events == 0 and
            not any(event.event_type == "kv_prefetch"
                    for event in zero_result.events),
            "zero prefetch_distance must disable active lookahead")
    return {
        "distance_tokens": 17,
        "lookahead_pages": details.get("lookahead_pages"),
        "prefetch_events": metrics.prefetch_events,
        "prefetch_bytes": metrics.prefetch_bytes,
        "prefetch_transfer_time_ns": metrics.prefetch_transfer_time_ns,
        "zero_distance_events": zero_result.kv_metrics.prefetch_events,
        "finished_requests": sum(
            1 for state in runtime.states.values()
            if state.status.value == "finished"
        ),
        "zero_runtime": zero_runtime.now,
    }


def check_hot_cold_session_scan() -> dict[str, Any]:
    """Compare one active slot with two active slots on the same trace."""

    requests = (
        RequestSpec(
            request_id="session-low", arrival_ns=0.0,
            prompt_tokens=32, output_tokens=8, priority=0,
            deadline_ns=1_000_000_000.0,
        ),
        RequestSpec(
            request_id="session-high", arrival_ns=1.0,
            prompt_tokens=32, output_tokens=8, priority=10,
            deadline_ns=1_000_000_000.0,
        ),
    )
    rows = []
    for active_slots in (1, 2):
        scenario = _serving_paged_scenario(
            hbf_capacity=4 * PAGE,
            hbm_capacity=8 * PAGE,
            requests=requests,
            prefetch_distance=17,
            max_num_seqs=active_slots,
        )
        plan = compile_serving_plan(scenario)
        runtime = _OnlineRuntime(
            plan, lambda *_: BatchCost(1.0),
        )
        result = runtime.run()
        request_metrics = tuple(result.request_metrics.values())
        finished_metrics = tuple(
            metric for metric in request_metrics
            if metric.status.value == "finished"
        )
        output_tokens = sum(
            int(metric.visible_output_tokens) for metric in finished_metrics
        )
        goodput_tokens = sum(
            int(metric.visible_output_tokens)
            for metric in finished_metrics
            if metric.deadline_met is True
        )
        makespan_s = max(1e-9, float(result.makespan_ns) / 1e9)
        ttft_values = sorted(
            float(metric.ttft_ns) for metric in finished_metrics
            if metric.ttft_ns is not None
        )
        tpot_values = sorted(
            float(metric.tpot_ns) for metric in finished_metrics
            if metric.tpot_ns is not None
        )
        rows.append({
            "active_slots": active_slots,
            "retained_session_budget": active_slots,
            "active_request_peak": runtime._max_batch_sequences,
            "preemptions": runtime.preemptions,
            "swap_events": runtime.swap_events,
            "prefetch_events": result.kv_metrics.prefetch_events,
            "finished_requests": len(finished_metrics),
            "rejected_requests": sum(
                1 for metric in request_metrics
                if metric.status.value == "rejected"
            ),
            "node_tps": output_tokens / makespan_s,
            "slo_goodput_tokens_per_s": goodput_tokens / makespan_s,
            "ttft_ns": ttft_values,
            "tpot_ns": tpot_values,
            "makespan_ns": result.makespan_ns,
        })
    require(rows[0]["preemptions"] >= rows[1]["preemptions"],
            "fewer active slots must not reduce preemption pressure")
    require(rows[0]["swap_events"] >= rows[1]["swap_events"],
            "fewer active slots must not reduce KV swap pressure")
    require(all(row["finished_requests"] == 2 for row in rows),
            "hot/cold session scan must finish the same request trace")
    require(all(row["rejected_requests"] == 0 and
                row["slo_goodput_tokens_per_s"] > 0.0
                for row in rows),
            "hot/cold session scan must report finite SLO goodput")
    return {
        "trace": "two requests with a one-nanosecond priority gap",
        "rows": rows,
        "interpretation": (
            "active slot count is the retained-hot-session budget; swapped pages "
            "represent cold sessions and their restore cost"
        ),
    }


def check_slo_semantics() -> dict[str, Any]:
    """SLO thresholds classify the same fixed trace without retiming it."""

    def run(deadline_ns: float):
        scenario = _serving_paged_scenario(
            hbf_capacity=16 * 1024 * 1024,
            requests=(RequestSpec(
                request_id="slo",
                arrival_ns=0.0,
                prompt_tokens=16,
                output_tokens=4,
                deadline_ns=deadline_ns,
            ),),
        )
        runtime = _OnlineRuntime(
            compile_serving_plan(scenario),
            lambda *_: BatchCost(100.0),
        )
        return runtime.run()

    loose = run(1_000_000.0)
    tight = run(1.0)
    loose_metric = loose.request_metrics["slo"]
    tight_metric = tight.request_metrics["slo"]
    require(equal(loose.makespan_ns, tight.makespan_ns),
            "changing an SLO threshold must not retime a fixed single-request trace")
    require(loose_metric.deadline_met is True and tight_metric.deadline_met is False,
            "SLO threshold must change qualification without changing execution")
    return {
        "makespan_ns": loose.makespan_ns,
        "loose_deadline_met": loose_metric.deadline_met,
        "tight_deadline_met": tight_metric.deadline_met,
        "loose_ttft_ns": loose_metric.ttft_ns,
        "tight_ttft_ns": tight_metric.ttft_ns,
    }


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise AssertionError(explanation)


def _load_acceptance_module(filename: str, module_name: str):
    """Load one standalone acceptance helper without making scripts a package."""
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load acceptance helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def check_core_invariants() -> dict[str, Any]:
    """Run hand-calculable, aggregate and bounded-event core checks."""
    module = _load_acceptance_module("hbf_hbm_invariants.py", "hbf_hbm_invariants_acceptance")
    report = module.run_invariants()
    require(report["passed"], "one or more direct HBM/HBF core invariants failed")
    return report


def check_package_budget() -> dict[str, Any]:
    """Run the coupled four-slot capacity/bandwidth/buffer budget sweep."""
    module = _load_acceptance_module("hbf_hbm_budget.py", "hbf_hbm_budget_acceptance")
    report = module.run_budget_sweep()
    require(report["calibrated_to_silicon"] is False,
            "package budget sweep must remain explicitly uncalibrated")
    return report


def check_session_lifecycle() -> dict[str, Any]:
    """Validate create/append/pause/resume/end on the physical KV pool."""
    module = _load_acceptance_module("hbf_hbm_sessions.py", "hbf_hbm_sessions_acceptance")
    report = module.run_matrix()
    rows = report["experiments"]
    require(len(rows) == 12, "session lifecycle matrix must cover all declared scan points")
    required = {"create", "append", "pause", "pause_complete", "resume_requested", "resume", "end"}
    for row in rows:
        event_types = set(row["event_types"])
        require(required.issubset(event_types),
                "every session trace must expose the full lifecycle event set")
        require(row["remaining_pages"] == 0 and not any(row["used_bytes_by_component"].values()),
                "ending all sessions must release every physical KV page")
    migrated = [row for row in rows if row["mode"] in {"all_hbf", "hot_cold"}]
    require(any(row["migration_count"] > 0 and row["restore_wait_ns"] > 0 for row in migrated),
            "cold-session traces must wait for physical restore completion")
    return {
        "scan_axes": report["scan_axes"],
        "rows": [
            {key: row[key] for key in (
                "mode", "pause_duration_ns", "resume_cycles", "active_sessions_peak",
                "retained_sessions_peak", "pause_count", "migration_count",
                "migration_bytes", "restore_wait_ns", "released_pages",
                "remaining_pages", "makespan_ns",
            )}
            for row in rows
        ],
        "limitations": report["limitations"],
    }


def equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8)


def hbm_config(**changes: Any) -> DramConfig:
    return resolve_physical_config("research_hbm_acceptance", overrides=changes)


def hbf_config(**changes: Any) -> NandConfig:
    return resolve_physical_config("research_hbf_sessions", overrides=changes)


def access(core: Any, operation: str, *, size: int = PAGE,
           address: int = 0, arrival_ns: float = 0.0, label: str = "access"):
    return core.submit(AccessRequest(label, operation, address, size, arrival_ns))


def check_directional_bandwidth() -> dict[str, Any]:
    fast, slow = hbf_config(), hbf_config(host_write_bandwidth_gb_s=13.6)
    observations = []
    for details, pages in ((True, 1), (True, 128), (False, 128)):
        row: dict[str, Any] = {"capture_details": details, "pages": pages}
        for operation in ("read", "write"):
            values = [access(NandCore(config, capture_details=details), operation,
                             size=pages * PAGE).completion_ns for config in (fast, slow)]
            row[operation + "_fast_ns"], row[operation + "_slow_ns"] = values
            require(equal(*values) if operation == "read" else values[1] > values[0],
                    "write-bandwidth change must affect writes and leave reads unchanged")
        observations.append(row)
    require(equal(observations[1]["write_fast_ns"], observations[2]["write_fast_ns"]),
            "detailed and accelerated NAND write timing disagree")
    return {"read_gb_s": 488.0, "write_gb_s": [27.2, 13.6], "observations": observations}


def check_standard_kv_bytes() -> dict[str, Any]:
    """Validate the hand-computable BF16 GQA KV byte baseline."""

    bytes_per_element = 2  # BF16
    layers = 80
    kv_heads = 8
    head_dim = 128
    tokens = 131_072
    bytes_per_token = 2 * layers * kv_heads * head_dim * bytes_per_element
    total_bytes = bytes_per_token * tokens
    require(bytes_per_token == 327_680,
            "standard BF16 GQA KV bytes/token baseline changed")
    require(total_bytes == 40 * 1024**3,
            "131072-token standard KV baseline must equal 40 GiB")
    return {
        "dtype": "bf16",
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "bytes_per_token": bytes_per_token,
        "tokens": tokens,
        "total_bytes": total_bytes,
        "total_gib": total_bytes / 1024**3,
    }


def check_write_buffer() -> dict[str, Any]:
    config = hbf_config(write_buffer_bytes=8192)
    results = [access_core for access_core in NandCore(config).run(
        AccessRequest(str(index), "write", index * PAGE, 4096, 0.0) for index in range(3)
    )]
    batch = summarize_batch(results)
    require(results[0].write_buffer_wait_ns == results[1].write_buffer_wait_ns == 0,
            "the first two writes must fit the finite buffer")
    require(results[2].write_buffer_wait_ns > 0, "buffer exhaustion must backpressure the third write")
    require(batch.write_buffer_peak_bytes == 8192, "write buffer peak must equal its declared capacity")
    require(batch.write_buffer_bytes == 12288, "write buffer byte accounting is incorrect")
    drained = access(NandCore(config), "write", size=4096,
                     arrival_ns=batch.last_completion_ns)
    require(drained.write_buffer_wait_ns == 0, "empty buffer must admit a later write")
    try:
        access(NandCore(config), "write", size=12288)
    except ValueError:
        pass
    else:
        raise AssertionError("an individually oversized buffered write must fail explicitly")
    buffer_scan = []
    for capacity in (PAGE // 4, PAGE // 2, PAGE):
        scan_results = [access_core for access_core in NandCore(
            hbf_config(write_buffer_bytes=capacity), capture_details=False
        ).run(
            AccessRequest(str(index), "write", index * 4096, 4096, 0.0)
            for index in range(8)
        )]
        scan_batch = summarize_batch(scan_results)
        require(scan_batch.write_buffer_peak_bytes == capacity,
                "write-buffer scan must reach each declared capacity")
        buffer_scan.append({
            "capacity_bytes": capacity,
            "wait_ns": scan_batch.write_buffer_wait_ns,
            "peak_bytes": scan_batch.write_buffer_peak_bytes,
            "logical_write_bytes": scan_batch.write_buffer_bytes,
        })
    require(buffer_scan[0]["wait_ns"] > buffer_scan[-1]["wait_ns"],
            "larger write buffers must reduce sustained backpressure")
    buffered_core = NandCore(
        hbf_config(
            write_buffer_bytes=2 * PAGE,
        ),
    )
    first = buffered_core.submit(
        AccessRequest("buffered-0", "write", 0, PAGE // 4, 0.0)
    )
    second = buffered_core.submit(
        AccessRequest("buffered-1", "write", PAGE // 4, PAGE // 4, 0.0)
    )
    staged_read = buffered_core.submit(
        AccessRequest("buffered-read", "read", 0, PAGE // 4, 0.0)
    )
    drain = buffered_core.flush_write_buffer()
    require(first.counters["write_completion"] == "host_buffer_ack_pending_media_drain",
            "buffered write must acknowledge at the host before media drain")
    require(second.counters["merge_bytes"] == PAGE // 4,
            "small writes to one page must report merged host bytes")
    require(staged_read.counters["staging_read_hits"] == 1
            and staged_read.physical_read_bytes == 0,
            "reads must use the staged page before media drain")
    require(drain["flushed_pages"] == 1 and drain["physical_programs"] == 1
            and drain["write_completion"] == "media_program_complete",
            "drain must program one merged page and report media completion")
    return {"buffer_capacity_bytes": 8192, "write_buffer_wait_ns": batch.write_buffer_wait_ns,
            "write_buffer_peak_bytes": batch.write_buffer_peak_bytes,
            "host_payload_bytes": batch.write_buffer_bytes, "oversized_request_rejected": True,
            "buffer_scan": buffer_scan,
            "buffered_semantics": {
                "merge_bytes": second.counters["merge_bytes"],
                "staging_read_hits": staged_read.counters["staging_read_hits"],
                "physical_programs_on_drain": drain["physical_programs"],
                "drain_completion_ns": drain["media_drain_completion_ns"],
            }}


def check_capacity_only() -> dict[str, Any]:
    # Keep geometry and bandwidth identical; change only addressable capacity.
    durations = []
    for capacity in (8 * PAGE, 64 * PAGE):
        core = NandCore(hbf_config(capacity_bytes=capacity), capture_details=False)
        durations.append(core.run(AccessRequest(str(i), "read", i * PAGE, PAGE)
                                  for i in range(8)).duration_ns)
    require(equal(*durations), "capacity alone must not accelerate an already-fitting working set")
    # An explicit ideal-link control contains neither media latency nor capacity
    # pressure, so identical bytes and bandwidth have identical service time.
    ideal = [ResourceTimeline().transfer("ideal", 8 * PAGE, 0.0, 488.0).duration_ns
             for _capacity in (8 * PAGE, 64 * PAGE)]
    require(equal(*ideal), "ideal fixed-workload bandwidth control changed")
    small = DynamicKVPool([KvPoolComponent("hbm", 2 * PAGE, kind="hbm")], page_bytes=PAGE)
    require(small.resize_detailed("request", 2).success, "HBM capacity should fit two pages")
    require(not small.resize_detailed("request", 3).success, "HBM capacity overflow must reject allocation")
    require(len(small.pages()) == 2 and small.used_bytes_by_component["hbm"] == 2 * PAGE,
            "failed capacity admission must not leak pages")
    extended = DynamicKVPool([
        KvPoolComponent("hbm", 2 * PAGE, kind="hbm"),
        KvPoolComponent("hbf", 8 * PAGE, kind="hbf", active=True,
                        read_bandwidth_gbps=488 * 8, write_bandwidth_gbps=27.2 * 8),
    ], page_bytes=PAGE)
    require(extended.resize_detailed("request", 10).success, "active-memory HBF must increase admissible resident pages")
    require(not extended.resize_detailed("request", 11).success, "combined capacity overflow must reject allocation")
    extended.release("request")
    require(sum(extended.used_bytes_by_component.values()) == 0, "release must reclaim both tiers")
    return {"fixed_working_set_bytes": 8 * PAGE, "capacities_bytes": [8 * PAGE, 64 * PAGE],
            "memory_service_duration_ns": durations, "ideal_link_duration_ns": ideal,
            "hbm_only_max_pages": 2, "hbm_plus_active_hbf_max_pages": 10,
            "capacity_overflow_rejected": True, "release_reclaims_all_bytes": True}


def check_shared_contention() -> dict[str, Any]:
    config = hbf_config()
    isolated = access(NandCore(config), "read").completion_ns
    shared = NandCore(config)
    first = access(shared, "read", label="foreground")
    second = access(shared, "read", label="background")
    require(equal(first.completion_ns, isolated), "first shared access must match isolated timing")
    require(second.completion_ns > isolated, "same physical resources must serialize conflicting reads")
    return {"isolated_read_completion_ns": isolated,
            "shared_first_completion_ns": first.completion_ns,
            "shared_second_completion_ns": second.completion_ns,
            "contention_delay_ns": second.completion_ns - isolated}


def _migration_hardware(*, relay: bool, storage: bool = False) -> HardwareSpec:
    """Build a tiny physical graph with explicit memory services and links."""

    hbm_capacity = 4 * PAGE
    hbf_capacity = 4 * PAGE
    hbm_physical = resolve_physical_config(
        "research_hbm_acceptance", overrides={"capacity_bytes": hbm_capacity})
    hbf_physical = resolve_physical_config("research_hbf_sessions", overrides={
        "capacity_bytes": hbf_capacity, "write_buffer_bytes": 2 * PAGE})
    components = [
        ComponentSpec(
            "hbm", "hbm", capacity_bytes=hbm_capacity,
            bandwidth_gbps=488.0 * 8.0,
            metadata={"physical_memory_config": hbm_physical},
        ),
        ComponentSpec(
            "hbf", "hbf", capacity_bytes=hbf_capacity,
            bandwidth_gbps=488.0 * 8.0,
            read_bandwidth_gbps=488.0 * 8.0,
            write_bandwidth_gbps=27.2 * 8.0,
            metadata={
                "physical_memory_config": hbf_physical,
                **(
                    {
                        "access_mode": "memory",
                        "write_buffer_bytes": 2 * PAGE,
                    }
                    if not storage
                    else {"access_mode": "remote_flash"}
                ),
            },
        ),
    ]
    links = []
    if relay:
        relay_capacity = 4 * PAGE
        relay_physical = resolve_physical_config(
            "research_hbm_relay", overrides={"capacity_bytes": relay_capacity})
        components.insert(
            1,
            ComponentSpec(
                "relay", "hbm", capacity_bytes=relay_capacity,
                bandwidth_gbps=256.0 * 8.0,
                metadata={"physical_memory_config": relay_physical},
            ),
        )
        links.extend([
            LinkSpec("hbm-relay", "hbm", "p0", "relay", "p0", "pcie",
                     bandwidth_gbps=64.0 * 8.0, latency_ns=50.0),
            LinkSpec("relay-hbf", "relay", "p1", "hbf", "p1", "pcie",
                     bandwidth_gbps=64.0 * 8.0, latency_ns=50.0),
        ])
    else:
        links.append(
            LinkSpec("hbm-hbf", "hbm", "p0", "hbf", "p0", "pcie",
                     bandwidth_gbps=64.0 * 8.0, latency_ns=50.0)
        )
    return HardwareSpec("kv-migration", tuple(components), tuple(links), require_connected=False)


def _run_physical_migration(*, relay: bool, storage: bool = False) -> dict[str, Any]:
    hardware = _migration_hardware(relay=relay, storage=storage)
    executor = PhysicalKVTransferExecutor(
        hardware.components,
        runtime=PhysicalRuntimeContext(),
        topology=TopologyRouter(hardware),
        relay_components=("relay",) if relay else (),
    )
    pool = DynamicKVPool(
        [
            KvPoolComponent("hbm", 4 * PAGE, kind="hbm"),
            KvPoolComponent(
                "hbf", 4 * PAGE, kind="hbf", active=not storage,
                read_bandwidth_gbps=488.0 * 8.0,
                write_bandwidth_gbps=27.2 * 8.0,
            ),
        ],
        page_bytes=PAGE,
    )
    pool.set_transfer_executor(executor)
    require(pool.resize_detailed("migration", 1, compatible_components=("hbm",)).success,
            "physical migration source allocation failed")
    executor.sync_pages(pool.pages())
    page = pool.pages()[0]
    source_address = executor.address(page)
    migrated = (
        pool.offload_page(page.logical_page_id, "hbf")
        if storage
        else pool.migrate_page(page.logical_page_id, "hbf")
    )
    require(migrated, "physical migration failed: " + str(pool.last_error))
    event = pool.events[-1]
    physical_event = executor.events[-1]
    phase_kinds = [str(phase.get("kind")) for phase in physical_event["phases"]]
    require(page.owner_component == "hbf", "migration did not update page ownership")
    # Address spaces are owner-scoped, so both allocators may legitimately
    # return base address zero.  The ownership transition plus a valid target
    # allocation is the invariant; comparing raw integers across owners would
    # be a false failure.
    require(page.owner_component == "hbf" and executor.address(page) >= 0,
            "migration must expose a valid target physical allocation")
    require("source_read" in phase_kinds and "target_write" in phase_kinds,
            "migration must include source read and target write phases")
    require("link" in phase_kinds, "migration must reserve a real fabric link")
    if relay:
        require("relay_write" in phase_kinds and "relay_read" in phase_kinds,
                "relay migration must include relay write/read phases")
    require(executor.counters["read_bytes"] >= PAGE and
            executor.counters["write_bytes"] >= PAGE,
            "physical migration counters must include media reads and writes")
    require(any(key.startswith("link.") for key in executor.runtime.timeline.bytes_moved),
            "migration must leave shared link bytes in the runtime timeline")
    return {
        "path": "relay" if relay else "direct",
        "target_tier": "storage" if storage else "active_memory",
        "logical_page_bytes": PAGE,
        "source_address": source_address,
        "target_address": executor.address(page),
        "duration_ns": event.duration_ns,
        "phase_kinds": phase_kinds,
        "read_bytes": executor.counters["read_bytes"],
        "write_bytes": executor.counters["write_bytes"],
        "link_bytes": executor.counters["link_bytes"],
        "timeline_bytes_moved": dict(executor.runtime.timeline.bytes_moved),
    }


def check_physical_kv_migration() -> dict[str, Any]:
    direct = _run_physical_migration(relay=False)
    relay = _run_physical_migration(relay=True)
    storage = _run_physical_migration(relay=False, storage=True)
    require(relay["duration_ns"] > direct["duration_ns"],
            "relay path must include additional physical work")
    require(storage["target_tier"] == "storage",
            "storage migration must exercise the explicit offload tier")
    return {"direct": direct, "relay": relay, "storage": storage}


def placement_experiment(name: str, *, kv_tier: str, weights_tier: str,
                         cold_pages: int = 0, kv_pages: int = 8,
                         steps: int = 4) -> dict[str, Any]:
    """Fixed synthetic decode-like memory steps; no compute/token model."""
    kv_pages = int(kv_pages)
    steps = int(steps)
    require(kv_pages >= 2 and steps > 0,
            "placement workload requires at least two KV pages and one step")
    cold_pages = max(0, min(int(cold_pages), kv_pages))
    cores = {"hbm": DramCore(hbm_config(), capture_details=False),
             "hbf": NandCore(hbf_config(write_buffer_bytes=2 * PAGE), capture_details=False)}
    by_tier: dict[str, list[Any]] = {"hbm": [], "hbf": []}
    clock = 0.0
    for step in range(steps):
        current = []
        # A weight scan followed by an independent KV batch. The next step
        # waits for all reads and appends, without assuming compute overlap.
        weight = access(cores[weights_tier], "read", size=8 * PAGE,
                        address=16 * PAGE, arrival_ns=clock, label=f"weights-{step}")
        by_tier[weights_tier].append(weight)
        current.append(weight)
        for page in range(kv_pages):
            tier = "hbf" if page < cold_pages else kv_tier
            result = access(cores[tier], "read", address=page * PAGE,
                            arrival_ns=clock, label=f"kv-read-{step}-{page}")
            by_tier[tier].append(result)
            current.append(result)
        for page in (kv_pages - 2, kv_pages - 1):
            result = access(cores[kv_tier], "write", address=page * PAGE,
                            arrival_ns=clock, label=f"kv-append-{step}-{page}")
            by_tier[kv_tier].append(result)
            current.append(result)
        clock = max(result.completion_ns for result in current)
    logical_bytes = sum(result.logical_bytes for values in by_tier.values() for result in values)
    tiers = {}
    for tier, values in by_tier.items():
        batch = summarize_batch(values)
        tiers[tier] = {"logical_read_bytes": sum(x.logical_bytes for x in values if x.operation is Operation.READ),
                       "logical_write_bytes": sum(x.logical_bytes for x in values if x.operation is Operation.WRITE),
                       "physical_read_bytes": batch.physical_read_bytes,
                       "physical_write_bytes": batch.physical_write_bytes,
                       "queue_wait_ns": batch.queue_wait_ns,
                       "write_buffer_wait_ns": batch.write_buffer_wait_ns,
                       "write_buffer_peak_bytes": batch.write_buffer_peak_bytes}
    return {"id": name, "kind": "fixed_traffic_memory_microbenchmark", "steps": steps,
            "kv_pages": kv_pages, "context_tokens": kv_pages * 16,
            "cold_pages_on_hbf": cold_pages,
            "weights_tier": weights_tier, "kv_default_tier": kv_tier,
            "memory_service_duration_ns": clock, "logical_bytes": logical_bytes,
            "useful_memory_bandwidth_gb_s": logical_bytes / clock,
            "tiers": tiers, "migration_count": 0,
            "placement_note": "Static placement; dynamic migration is tested separately."}


def check_fixed_workload_scan() -> dict[str, Any]:
    """Compare HBM-only and HBF-backed service for several fixed contexts."""

    rows = []
    for kv_pages in (2, 4, 8):
        hbm = placement_experiment(
            f"scan_{kv_pages}_hbm", kv_tier="hbm", weights_tier="hbm",
            kv_pages=kv_pages, steps=2,
        )
        hbf = placement_experiment(
            f"scan_{kv_pages}_hbf", kv_tier="hbf", weights_tier="hbm",
            kv_pages=kv_pages, steps=2,
        )
        require(hbm["logical_bytes"] == hbf["logical_bytes"],
                "context scan must preserve fixed logical traffic")
        require(hbf["memory_service_duration_ns"] >= hbm["memory_service_duration_ns"],
                "placing the same KV traffic on HBF must not be faster than HBM")
        rows.append({
            "context_tokens": kv_pages * 16,
            "kv_pages": kv_pages,
            "logical_bytes": hbm["logical_bytes"],
            "hbm_only_duration_ns": hbm["memory_service_duration_ns"],
            "all_kv_hbf_duration_ns": hbf["memory_service_duration_ns"],
            "hbm_only_useful_bandwidth_gb_s": hbm["useful_memory_bandwidth_gb_s"],
            "all_kv_hbf_useful_bandwidth_gb_s": hbf["useful_memory_bandwidth_gb_s"],
        })
    return {
        "contexts": rows,
        "fixed_workload": True,
        "scan_axes": ["context_tokens", "kv_tier"],
        "interpretation": (
            "Only context size and KV tier vary; model compute and request scheduling "
            "are outside this memory-traffic scan."
        ),
    }


def hbm_hbf_ratio_experiment(
    name: str, *, hbm_pages: int, hbf_pages: int,
    weight_pages: int = 4, kv_pages: int = 8,
) -> dict[str, Any]:
    """Run the same fixed traffic with a different HBM/HBF capacity split.

    The workload has four weight pages and eight KV pages.  Pages are packed
    into HBM first and spill to HBF when the selected HBM capacity is full.
    Both tiers use their declared physical capacity; addresses are local to
    each tier so a split never accidentally aliases an out-of-range address.
    This is intentionally a small analytical ratio sweep, rather than a
    product-level packaging model.
    """
    hbm_pages = int(hbm_pages)
    hbf_pages = int(hbf_pages)
    weight_pages = int(weight_pages)
    kv_pages = int(kv_pages)
    require(hbm_pages > 0 and hbf_pages > 0,
            "HBM/HBF ratio must allocate at least one page to each tier")
    require(weight_pages > 0 and kv_pages > 0,
            "ratio workload must contain positive weight and KV pages")
    total_pages = weight_pages + kv_pages
    require(hbm_pages + hbf_pages >= total_pages,
            "ratio capacity must fit the fixed weight plus KV working set")

    hbm_core = DramCore(
        hbm_config(capacity_bytes=hbm_pages * PAGE), capture_details=False
    )
    # Keep a finite buffer in every ratio while ensuring it fits the smallest
    # HBF slice.  This makes the write-path pressure part of the comparison.
    hbf_core = NandCore(
        hbf_config(
            capacity_bytes=hbf_pages * PAGE,
            write_buffer_bytes=min(2 * PAGE, hbf_pages * PAGE),
        ),
        capture_details=False,
    )
    cores = {"hbm": hbm_core, "hbf": hbf_core}

    # Allocate a deterministic page map.  HBM is preferred for the working
    # set and HBF receives the remainder, so changing the ratio changes only
    # residency and the resulting resource service time.
    hbm_remaining = hbm_pages
    assignment: list[tuple[str, int]] = []
    local_offsets = {"hbm": 0, "hbf": 0}
    for _index in range(total_pages):
        tier = "hbm" if hbm_remaining > 0 else "hbf"
        hbm_remaining -= int(tier == "hbm")
        local_address = local_offsets[tier] * PAGE
        local_offsets[tier] += 1
        assignment.append((tier, local_address))
    weight_assignment = assignment[:weight_pages]
    kv_assignment = assignment[weight_pages:]

    by_tier: dict[str, list[Any]] = {"hbm": [], "hbf": []}
    clock = 0.0
    for step in range(4):
        current = []
        for page, (tier, address) in enumerate(weight_assignment):
            result = access(
                cores[tier], "read", address=address, arrival_ns=clock,
                label=f"ratio-weight-{step}-{page}",
            )
            by_tier[tier].append(result)
            current.append(result)
        for page, (tier, address) in enumerate(kv_assignment):
            result = access(
                cores[tier], "read", address=address, arrival_ns=clock,
                label=f"ratio-kv-read-{step}-{page}",
            )
            by_tier[tier].append(result)
            current.append(result)
        # Two KV append pages follow the reads, preserving the same traffic
        # shape for each ratio and exposing HBF's slower write direction.
        for page in (6, 7):
            tier, address = kv_assignment[page]
            result = access(
                cores[tier], "write", address=address, arrival_ns=clock,
                label=f"ratio-kv-append-{step}-{page}",
            )
            by_tier[tier].append(result)
            current.append(result)
        clock = max(result.completion_ns for result in current)

    logical_bytes = sum(
        result.logical_bytes
        for values in by_tier.values()
        for result in values
    )
    tiers: dict[str, dict[str, Any]] = {}
    for tier, values in by_tier.items():
        batch = summarize_batch(values)
        tiers[tier] = {
            "logical_read_bytes": sum(
                x.logical_bytes for x in values if x.operation is Operation.READ
            ),
            "logical_write_bytes": sum(
                x.logical_bytes for x in values if x.operation is Operation.WRITE
            ),
            "physical_read_bytes": batch.physical_read_bytes,
            "physical_write_bytes": batch.physical_write_bytes,
            "queue_wait_ns": batch.queue_wait_ns,
            "write_buffer_wait_ns": batch.write_buffer_wait_ns,
            "write_buffer_peak_bytes": batch.write_buffer_peak_bytes,
        }
    hbm_resident_pages = sum(1 for tier, _address in assignment if tier == "hbm")
    hbf_resident_pages = sum(1 for tier, _address in assignment if tier == "hbf")
    return {
        "id": name,
        "kind": "fixed_traffic_hbm_hbf_ratio",
        "steps": 4,
        "working_set_pages": total_pages,
        "weight_pages": weight_pages,
        "kv_pages": kv_pages,
        "hbm_capacity_pages": hbm_pages,
        "hbf_capacity_pages": hbf_pages,
        "hbm_capacity_ratio": hbm_pages / (hbm_pages + hbf_pages),
        "hbm_resident_pages": hbm_resident_pages,
        "hbf_resident_pages": hbf_resident_pages,
        "memory_service_duration_ns": clock,
        "logical_bytes": logical_bytes,
        "useful_memory_bandwidth_gb_s": logical_bytes / clock,
        "tiers": tiers,
        "migration_count": 0,
        "ratio_note": (
            "Fixed 12-page working set; HBM is filled first and the remainder "
            "is served by HBF."
        ),
    }


def check_hbm_hbf_ratio() -> dict[str, Any]:
    """Scan capacity splits and verify capacity/traffic semantics."""
    rows = [
        hbm_hbf_ratio_experiment(
            f"ratio_hbm_{hbm_pages}_hbf_{12 - hbm_pages}",
            hbm_pages=hbm_pages,
            hbf_pages=12 - hbm_pages,
        )
        for hbm_pages in (2, 4, 8, 10)
    ]
    logical_bytes = {row["logical_bytes"] for row in rows}
    require(len(logical_bytes) == 1,
            "changing capacity ratio must not change fixed traffic volume")
    durations = [float(row["memory_service_duration_ns"]) for row in rows]
    require(all(duration > 0.0 for duration in durations),
            "every ratio must produce a positive service duration")
    # More HBM should reduce the amount of traffic sent through HBF's slower
    # write path for this fixed working set.  Keep the check directional
    # rather than asserting a particular silicon-calibrated speedup.
    require(durations[-1] < durations[0],
            "moving capacity from HBF to HBM must reduce service time")
    return {
        "fixed_working_set_pages": 12,
        "rows": rows,
        "logical_bytes_equal": True,
        "duration_ns": durations,
        "best_ratio": rows[durations.index(min(durations))]["id"],
    }


def check_backend_policy_sensitivity() -> dict[str, Any]:
    """Verify non-memory controls change execution through the real backend."""
    audit_spec = importlib.util.spec_from_file_location(
        "hbf_hbm_core_audit_acceptance", ROOT / "scripts" / "hbf_hbm_core_audit.py"
    )
    audit_module = importlib.util.module_from_spec(audit_spec)
    assert audit_spec.loader is not None
    audit_spec.loader.exec_module(audit_module)
    script = ROOT / "scripts" / "calibrated_hbf_profiles.py"
    spec = importlib.util.spec_from_file_location("calibrated_hbf_profiles_acceptance", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    profile = module._profile_by_id("son_hbf")
    with audit_module.CoreExecutionAudit() as audit:
        baseline = module._full_backend_summary(
            profile, use_hbf=False, request_count=1, prompt_tokens=32, output_tokens=4
        )
    core_observations = audit.summary()
    live_hbm = [row for row in core_observations["owners"] if row["memory_kind"] == "HBM"]
    require(core_observations["valid_for_committed_byte_totals"],
            "default HBM evidence must come from successful persistent Core submissions")
    require(bool(live_hbm) and all(row["core"] == "DramCore" for row in live_hbm),
            "default HBM operators must actually execute through DramCore")
    require(sum(row["logical_read_bytes"] for row in live_hbm) > 0
            and sum(row["logical_write_bytes"] for row in live_hbm) > 0
            and sum(row["burst_count"] for row in live_hbm) > 0,
            "default HBM must execute physical read and write bursts")
    gpu = module._full_backend_summary(
        profile, use_hbf=False, request_count=1, prompt_tokens=32, output_tokens=4,
        gpu_compute_pflops=0.06,
    )
    host = module._full_backend_summary(
        profile, use_hbf=False, request_count=1, prompt_tokens=32, output_tokens=4,
        host_control_scale=10.0,
    )
    link = module._full_backend_summary(
        profile, use_hbf=True, request_count=1, prompt_tokens=32, output_tokens=4,
        interconnect_bandwidth_gbps=8.0,
    )
    baseline_tensor_busy = baseline["resources"]["gpu_resource_busy_ns"].get("gpu0.tensor_core", 0.0)
    override_tensor_busy = gpu["resources"]["gpu_resource_busy_ns"].get("gpu0.tensor_core", 0.0)
    require(baseline["resources"]["gpu_busy_recorded_batches"] > 0
            and gpu["resources"]["gpu_busy_recorded_batches"] > 0,
            "GPU occupancy must come from actual persistent-live-kernel batches")
    require(override_tensor_busy > baseline_tensor_busy > 0.0,
            "slower GPU compute must increase actually executed tensor-core occupancy")
    # Memory and compute overlap. A slower GPU can consume more compute time
    # without changing makespan while physical HBM service remains limiting.
    require(gpu["summary"]["makespan_ns"] >= baseline["summary"]["makespan_ns"],
            "slower GPU compute must not shorten the fixed full-backend trace")
    require(gpu["resources"]["dram_traffic"]["logical_read_bytes"] == baseline["resources"]["dram_traffic"]["logical_read_bytes"],
            "GPU compute override must not change logical DRAM traffic")
    require(host["summary"]["makespan_ns"] > baseline["summary"]["makespan_ns"],
            "host control scaling must increase full-backend makespan")
    require(link["summary"]["makespan_ns"] > module._full_backend_summary(
        profile, use_hbf=True, request_count=1, prompt_tokens=32, output_tokens=4
    )["summary"]["makespan_ns"], "link bandwidth override must affect HBF makespan")
    return {
        "hbm_backend": "DramCore",
        "gpu_reference_makespan_ns": baseline["summary"]["makespan_ns"],
        "gpu_override_makespan_ns": gpu["summary"]["makespan_ns"],
        "gpu_reference_tensor_busy_ns": baseline_tensor_busy,
        "gpu_override_tensor_busy_ns": override_tensor_busy,
        "gpu_makespan_unchanged": gpu["summary"]["makespan_ns"] == baseline["summary"]["makespan_ns"],
        "gpu_check_scope": "actual tensor-core occupancy must increase; overlapped HBM service can leave total makespan unchanged",
        "gpu_busy_scope": baseline["resources"]["gpu_busy_scope"],
        "host_scaled_makespan_ns": host["summary"]["makespan_ns"],
        "hbf_slow_link_makespan_ns": link["summary"]["makespan_ns"],
        "live_hbm_core_observations": live_hbm,
        "physical_dram_task_count": baseline["resources"]["dram_traffic"]["task_count"],
    }


def run_acceptance() -> dict[str, Any]:
    checks = []
    tasks: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("directional_bandwidth", check_directional_bandwidth),
        ("standard_kv_bytes", check_standard_kv_bytes),
        ("finite_write_buffer", check_write_buffer),
        ("capacity_only_and_lifecycle", check_capacity_only),
        ("shared_physical_contention", check_shared_contention),
        ("physical_kv_migration", check_physical_kv_migration),
        ("core_invariants", check_core_invariants),
        ("hbm_hbf_ratio", check_hbm_hbf_ratio),
        ("backend_policy_sensitivity", check_backend_policy_sensitivity),
        ("package_budget_sweep", check_package_budget),
        ("fixed_workload_scan", check_fixed_workload_scan),
        ("serving_paged_swap", check_serving_paged_swap),
        ("prefetch_distance", check_prefetch_distance),
        ("hot_cold_session_scan", check_hot_cold_session_scan),
        ("session_lifecycle", check_session_lifecycle),
        ("slo_semantics", check_slo_semantics),
    ]
    for name, function in tasks:
        try:
            checks.append({"id": name, "passed": True, "metrics": function()})
        except Exception as error:
            checks.append({"id": name, "passed": False,
                           "error": f"{type(error).__name__}: {error}"})
    experiments = []
    for name, options in (
        ("A_hbm_only", {"kv_tier": "hbm", "weights_tier": "hbm"}),
        ("B_all_kv_hbf", {"kv_tier": "hbf", "weights_tier": "hbm"}),
        ("C_hot_hbm_cold_hbf", {"kv_tier": "hbm", "weights_tier": "hbm", "cold_pages": 4}),
        ("D_weights_hbf_kv_hbm", {"kv_tier": "hbm", "weights_tier": "hbf"}),
    ):
        try:
            experiments.append({"passed": True, **placement_experiment(name, **options)})
        except Exception as error:
            experiments.append({"id": name, "passed": False,
                                "error": f"{type(error).__name__}: {error}"})
    passed = {item["id"]: item["passed"] for item in checks}
    capabilities = [
        {"question": "Does the declared HBF write bandwidth change physical write service time?",
         "supported": passed["directional_bandwidth"], "evidence": ["directional_bandwidth"]},
        {"question": "When does a finite HBF write buffer introduce backpressure?",
         "supported": passed["finite_write_buffer"], "evidence": ["finite_write_buffer"]},
        {"question": "Does the standard BF16 GQA KV byte baseline match the hand calculation?",
         "supported": passed["standard_kv_bytes"], "evidence": ["standard_kv_bytes"]},
        {"question": "Does HBF expand capacity without automatically speeding up a fixed fitting workload?",
         "supported": passed["capacity_only_and_lifecycle"], "evidence": ["capacity_only_and_lifecycle"]},
        {"question": "Which HBM/HBF capacity split best serves the same fixed working set under the modeled directional bandwidths?",
         "supported": passed["hbm_hbf_ratio"], "evidence": ["hbm_hbf_ratio"],
         "scope": "fixed 12-page traffic; capacity ratio only, no packaging budget"},
        {"question": "How does the same fixed KV traffic change from short to long contexts when placed in HBM or HBF?",
         "supported": passed["fixed_workload_scan"], "evidence": ["fixed_workload_scan"],
         "scope": "2/4/8 pages; memory service only"},
        {"question": "How does static KV/weight placement change fixed memory-traffic service time?",
         "supported": all(item["passed"] for item in experiments), "evidence": [item["id"] for item in experiments]},
        {"question": "Does a KV migration reserve source read, fabric transfer and target write on shared physical state?",
         "supported": passed["physical_kv_migration"], "evidence": ["physical_kv_migration"]},
        {"question": "Do ideal read, aggregate-vs-split timing, and long-context event-bound invariants hold?",
         "supported": passed["core_invariants"], "evidence": ["core_invariants"]},
        {"question": "Under a fixed four-slot package budget, how do HBM/HBF capacity, bandwidth, parallelism and write buffering trade off?",
         "supported": passed["package_budget_sweep"], "evidence": ["package_budget_sweep"],
         "scope": "hypothetical four-slot budget; no commercial packaging calibration"},
        {"question": "Can serving automatically preempt, swap a multi-page KV working set to HBF, resume it, and release it?",
         "supported": passed["serving_paged_swap"], "evidence": ["serving_paged_swap"]},
        {"question": "How does a scheduler active-slot limit change priority-preemption and swap pressure?",
         "supported": passed["hot_cold_session_scan"], "evidence": ["hot_cold_session_scan"],
         "scope": "two-request synthetic trace; this is a slot-pressure check, not a session lifecycle trace"},
        {"question": "What migration and restore costs arise from explicit create/append/pause/resume/end session events?",
         "supported": passed["session_lifecycle"], "evidence": ["session_lifecycle"],
         "scope": "two-session physical KV trace below the serving scheduler"},
        {"question": "Do SLO thresholds classify a fixed serving trace without changing its execution timeline?",
         "supported": passed["slo_semantics"], "evidence": ["slo_semantics"]},
        {"question": "Can serving execute finite page-level fixed-distance KV prefetch and measure its cost?",
         "supported": passed["prefetch_distance"],
         "evidence": ["prefetch_distance"],
         "scope": "fixed token lookahead with page-level serving transfers; no overlap benefit claim"},
        {"question": "What is the real LLM token/s improvement, TTFT or TPOT on a commercial HBF device?",
         "supported": False, "reason": "The serving check uses a tiny synthetic scheduler but has no full model compute or commercial-HBF calibration."},
    ]
    validation_matrix = [
        {
            "question": item["question"],
            "supported": bool(item["supported"]),
            "evidence": list(item.get("evidence", ())),
            **({"scope": item["scope"]} if "scope" in item else {}),
            **({"reason": item["reason"]} if "reason" in item else {}),
        }
        for item in capabilities
    ]
    plan_completion = [
        {
            "stage": "0_baseline",
            "status": "passed",
            "scope": "analytical KV byte-count and capacity/lifecycle mechanism checks, not GPU timing calibration",
            "evidence": ["standard_kv_bytes", "capacity_only_and_lifecycle"],
        },
        {
            "stage": "1_parameters_and_timing",
            "status": "passed",
            "scope": "directional bandwidth and shared resource timing",
            "evidence": ["directional_bandwidth", "shared_physical_contention"],
        },
        {
            "stage": "2_simplified_hardware_execution",
            "status": "passed",
            "scope": "simplified DRAM/NAND execution, transfer accounting and policy sensitivity",
            "evidence": ["shared_physical_contention", "physical_kv_migration", "core_invariants", "backend_policy_sensitivity"],
        },
        {
            "stage": "3_capacity_lifecycle_and_write_buffer",
            "status": "passed",
            "scope": "finite capacity, page lifetime and explicit host-ack/media-drain buffer behavior",
            "evidence": ["capacity_only_and_lifecycle", "finite_write_buffer"],
        },
        {
            "stage": "4_tiering_migration_and_prefetch",
            "status": "passed",
            "scope": "synthetic lifecycle, page transfers, fixed lookahead and SLO classification; no predictor-accuracy claim",
            "evidence": ["serving_paged_swap", "prefetch_distance", "session_lifecycle", "slo_semantics"],
        },
        {
            "stage": "5_metrics_and_experiment_entry",
            "status": "partial",
            "scope": "memory-service scans and experiment entry points; does not certify all attachment section 6.3 metrics",
            "evidence": ["fixed_workload_scan", "hbm_hbf_ratio", "package_budget_sweep"],
            "remaining": ["completion_assessment_by_research_runner"],
            "not_assessed_here": ["complete_section_6_3_metric_coverage", "representative_tail_latency_and_stable_goodput", "serving_session_capacity", "predictor_and_speculative_acceptance_metrics", "declared_assumption_cost_area_lifetime_studies"],
        },
        {
            "stage": "6_acceptance_and_question_matrix",
            "status": "partial",
            "scope": "backend checks only; the full research workload and architecture matrix is assessed by separate runners",
            "evidence": [item["id"] for item in checks],
            "remaining": ["completion_assessment_by_research_runner", "external_named_GPU_HBM_validation", "paper_trace_level_comparison"],
            "not_assessed_here": ["named_model_workload_matrix", "matched_ABCD_and_host_SSD_CXL_serving_comparison", "multi_GPU_and_common_budget_comparison", "full_requested_parameter_scans"],
            "external_validation": "pending_external_measurements",
        },
    ]
    # A failed prerequisite must never leave a stage labelled "passed".
    for stage in plan_completion:
        if not all(passed.get(key, False) for key in stage["evidence"]):
            stage["status"] = "failed"
    return {"schema_version": 2, "passed": all(item["passed"] for item in checks + experiments),
            "scope": "single-device synthetic HBM/HBF physical and serving acceptance",
            "assumptions": {"page_bytes": PAGE, "hbm_interface_gb_s": 488.0,
                            "hbf_host_read_gb_s": 488.0, "hbf_host_write_gb_s": 27.2,
                            "hbf_read_ns": 4000.0, "hbf_program_ns": 75000.0,
                            "hbf_channels_and_parallel_units": 8,
                            "calibrated_to_silicon": False},
            "baseline_validation": {
                "status": "not_available",
                "predicted": None,
                "measured": None,
                "relative_error": None,
                "reason": "No independent named-GPU/HBM measurement set was supplied; synthetic backend timing is reported without an invented error value.",
            },
            "checks": checks, "experiments": experiments, "capabilities": capabilities,
            "validation_matrix": validation_matrix,
            "plan_completion_scope": "Backend work packages only. A passed stage certifies its explicit scope and evidence, not the full research proposal or commercial hardware accuracy.",
            "plan_completion": plan_completion,
            "plan_complete": all(item["status"] == "passed" for item in plan_completion),
            "research_followups": {
                "assessed_by_research_runner": ["full model/workload/architecture comparison matrix", "complete metrics and parameter sensitivity outputs with explicit assumptions"],
                "external_evidence": ["independent named-GPU/HBM measurements", "real HBF silicon calibration", "paper-specific route and workload traces when obtainable"],
                "long_term_research": ["stateful FTL/GC, wear leveling and retention", "RAS, security and ECC", "controller QoS and base-die PIM"],
                "priority_semantics": "Long-term research is disclosed separately; it is not a prerequisite for passing the current simplified-backend checks or completing a clearly scoped simulation study.",
            },
            "external_validation": "pending_external_measurements",
            "limitations": ["Serving coverage uses a tiny synthetic scheduler workload; it does not claim calibrated LLM token throughput.",
                            "HBM memory accesses execute DramCore transactions; GPU compute and launch overhead retain the existing analytical operator/resource models, not cycle-accurate GPU execution.",
                            "Prefetch is modeled as fixed token lookahead rounded to pages in paged-pool serving; no predictive policy is included.",
                            "Positive write_buffer_bytes enables page merging, host acknowledgement and media drain; zero selects write-through.",
                            "The four-slot package sweep is a coupled hypothetical budget, not commercial packaging calibration.",
                            "No FTL/GC, wear, ECC or multi-GPU model."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also save the JSON report to this file.")
    arguments = parser.parse_args()
    report = run_acceptance()
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
