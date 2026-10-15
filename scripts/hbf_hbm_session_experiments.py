"""Group 04: physical session lifecycle and prefix-sharing experiments.

These are memory/lifecycle simulations, not LLM throughput benchmarks. B
allocates, reads and appends directly in an explicitly memory-accessible HBF
tier. C moves private cold pages on pause and restores them before reading.
All architectures receive the same offered lifecycle for each parameter row.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from math import ceil
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hbf_hbm_sessions import PAGE, TOKENS_PER_PAGE, _hardware, _executed_core_configurations
from heterollm_sim.communication import TopologyRouter
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.kv_pool import DynamicKVPool, KvPoolComponent
from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor


class SessionExperiment:
    def __init__(self, architecture: str, *, active_ratio: float = 0.2,
                 pause_ns: float = 100_000.0, cycles: int = 2,
                 prefix_reuse_rate: float = 0.0, hbm_pages: int = 12,
                 sessions: int = 10) -> None:
        if architecture not in {"A_hbm_only", "B_all_hbf", "C_hot_cold"}:
            raise ValueError("unknown session architecture")
        if sessions < 1 or not 0 <= prefix_reuse_rate <= 1:
            raise ValueError("sessions must be positive and prefix_reuse_rate within [0, 1]")
        self.architecture = architecture
        self.parameters = dict(active_ratio=active_ratio, pause_ns=pause_ns,
                               cycles=cycles, prefix_reuse_rate=prefix_reuse_rate,
                               hbm_pages=hbm_pages, hbf_pages=64, sessions=sessions,
                               pages_per_session=4, prefix_pages=2,
                               append_bytes=PAGE // TOKENS_PER_PAGE)
        self.clock = 0.0
        hardware = _hardware(hbm_pages=hbm_pages, hbf_pages=64)
        if architecture == "B_all_hbf":
            # This is an explicit architecture assumption, not a claim that
            # the flash tier of an ordinary offload system is GPU-addressable.
            hardware = replace(hardware, components=tuple(
                replace(item, metadata={**item.metadata, "access_mode": "memory",
                                        "write_buffer_bytes": 0})
                if item.component_id == "hbf" else item
                for item in hardware.components))
        self.runtime = PhysicalRuntimeContext()
        router = TopologyRouter(hardware)
        self.executor = PhysicalKVTransferExecutor(
            hardware.components, runtime=self.runtime, topology=router,
            arrival_ns=lambda _: self.clock,
            namespace="group04:" + architecture,
        )
        included = {"hbm"} if architecture == "A_hbm_only" else (
            {"hbf"} if architecture == "B_all_hbf" else {"hbm", "hbf"})
        self.pool = DynamicKVPool(
            [KvPoolComponent.from_component(item) for item in hardware.components
             if item.component_id in included],
            tokens_per_page=TOKENS_PER_PAGE, page_bytes=PAGE, topology=router,
            offload_components=("hbf",) if architecture == "C_hot_cold" else (),
            transfer_executor=self.executor,
        )
        self.active: set[str] = set()
        self.retained: set[str] = set()
        self.admitted: set[str] = set()
        self.rejected: set[str] = set()
        self.events: list[dict[str, Any]] = []
        self.pause_completed: dict[str, float] = {}
        self.resume_waits: list[float] = []
        self.actual_pause_ns: list[float] = []
        self.peak_bytes = {key: 0 for key in included}
        self.peak_with_copy = dict(self.peak_bytes)
        self.peak_active = self.peak_retained = self.peak_live = 0
        self.prefix_hits = 0
        self.prefix_offers = 0
        self.prefix_registered = False
        self.read_bytes = {key: 0 for key in included}
        self.append_bytes = dict(self.read_bytes)
        self.media_write_bytes = dict(self.read_bytes)
        self.copy_peak = 0
        self.released_sessions = 0

    def record(self, event: str, session: str, requested: float, **details: Any) -> None:
        used = dict(self.pool.used_bytes_by_component)
        for key in self.peak_bytes:
            self.peak_bytes[key] = max(self.peak_bytes[key], used.get(key, 0))
            self.peak_with_copy[key] = max(self.peak_with_copy[key], used.get(key, 0))
        self.peak_active = max(self.peak_active, len(self.active))
        self.peak_retained = max(self.peak_retained, len(self.retained))
        self.peak_live = max(self.peak_live, len(self.active | self.retained))
        self.events.append(dict(event_type=event, session_id=session,
                                requested_ns=requested, completion_ns=self.clock,
                                active=len(self.active), retained=len(self.retained),
                                used_bytes=used, **details))

    def access(self, page: Any, operation: str, *, append: bool = False) -> None:
        owner = page.owner_component
        amount = self.parameters["append_bytes"] if append else page.bytes
        result = (self.executor.read_page if operation == "read" else self.executor.write_page)(
            page, arrival_ns=self.clock, size_bytes=amount,
            offset_bytes=page.bytes - amount if append else 0,
        )
        start = self.clock
        self.clock = max(self.clock, float(result["completion_ns"]))
        if operation == "read":
            self.read_bytes[owner] += amount
        elif append:
            self.append_bytes[owner] += amount
        self.media_write_bytes[owner] += int(result.get("physical_write_bytes", 0))
        self.record("append" if append else operation, page.request_id, start,
                    component=owner, bytes=amount,
                    physical_write_bytes=result.get("physical_write_bytes", 0))

    def create(self, session: str, requested: float, reuse: bool) -> None:
        self.clock = max(self.clock, requested)
        self.prefix_offers += int(reuse)
        old_ids = {page.logical_page_id for page in self.pool.pages()}
        success = self.pool.resize_detailed(session, self.parameters["pages_per_session"],
                                            prefix_key="shared" if reuse else None).success
        if not success:
            self.rejected.add(session)
            self.record("create", session, requested, accepted=False,
                        error=self.pool.last_error)
            return
        self.executor.sync_pages(self.pool.pages())
        self.admitted.add(session)
        self.active.add(session)
        if reuse:
            self.prefix_hits += 1
        for page in self.pool.request_pages(session):
            if page.logical_page_id not in old_ids:
                self.access(page, "write")
        if not self.prefix_registered and self.parameters["prefix_reuse_rate"] > 0:
            self.pool.register_prefix("shared", self.pool.request_pages(session)[:2])
            self.prefix_registered = True
        self.record("create", session, requested, accepted=True, prefix_hit=reuse)
        self.active_access(session)

    def active_access(self, session: str) -> None:
        for page in self.pool.request_pages(session):
            self.access(page, "read")
        # Prefix pages are immutable: append only to the private final page.
        self.access(self.pool.request_pages(session)[-1], "write", append=True)

    def move(self, page: Any, target: str) -> None:
        before = dict(self.pool.used_bytes_by_component)
        source = page.owner_component
        move = self.pool.offload_page if target == "hbf" else self.pool.restore_page
        if not move(page.logical_page_id, target):
            raise RuntimeError(self.pool.last_error or "session migration failed")
        transfer = self.pool.events[-1]
        # Sequential migration keeps the original until the target finishes.
        # Count both copies during that interval, per physical component.
        for component in self.peak_with_copy:
            occupied = before.get(component, 0) + (page.bytes if component == target else 0)
            self.peak_with_copy[component] = max(self.peak_with_copy[component], occupied)
        self.copy_peak = max(self.copy_peak, page.bytes)
        self.clock = max(self.clock, float(transfer.completion_ns))
        self.record("migration", page.request_id, float(transfer.start_ns),
                    source=source, target=target, bytes=page.bytes)

    def pause(self, session: str, requested: float) -> None:
        if session not in self.active:
            return
        self.clock = max(self.clock, requested)
        if self.architecture == "C_hot_cold":
            for page in self.pool.request_pages(session):
                # A shared prefix stays in HBM for potential active readers;
                # only private cold state is eligible for migration.
                if page.prefix_key is None and page.owner_component == "hbm":
                    self.move(page, "hbf")
        self.active.remove(session)
        self.retained.add(session)
        self.pause_completed[session] = self.clock
        self.record("pause", session, requested)

    def resume(self, session: str, requested: float) -> None:
        if session not in self.retained:
            return
        self.clock = max(self.clock, requested)
        self.actual_pause_ns.append(self.clock - self.pause_completed[session])
        if self.architecture == "C_hot_cold":
            for page in self.pool.request_pages(session):
                if page.owner_component == "hbf":
                    self.move(page, "hbm")
        self.retained.remove(session)
        self.active.add(session)
        self.resume_waits.append(self.clock - requested)
        self.record("resume", session, requested, restore_wait_ns=self.clock - requested)
        self.active_access(session)

    def run(self) -> dict[str, Any]:
        p = self.parameters
        width = max(1, ceil(p["sessions"] * p["active_ratio"]))
        ids = [f"s{i:02d}" for i in range(p["sessions"])]
        groups = [ids[i:i + width] for i in range(0, len(ids), width)]
        reuse_count = min(len(ids) - 1, round(p["prefix_reuse_rate"] * len(ids)))
        # Fixed requested times are independent of simulated architecture.
        # The observed time can be later when memory service falls behind.
        for gi, group in enumerate(groups):
            t = gi * 10_000_000.0
            for session in group:
                index = ids.index(session)
                self.create(session, t, 0 < index <= reuse_count)
            for session in group:
                self.pause(session, t + 1_000_000.0)
        init_end = len(groups) * 10_000_000.0
        for cycle in range(p["cycles"]):
            for gi, group in enumerate(groups):
                t = init_end + cycle * (len(groups) * 10_000_000.0 + p["pause_ns"])
                t += gi * 10_000_000.0 + p["pause_ns"]
                for session in group:
                    self.resume(session, t)
                for session in group:
                    self.pause(session, t + 1_000_000.0)
        for session in sorted(self.admitted):
            self.pool.release(session)
            self.active.discard(session)
            self.retained.discard(session)
            self.executor.sync_pages(self.pool.pages())
            self.released_sessions += 1
            self.record("end", session, self.clock)
        if self.prefix_registered:
            self.pool.release_prefix("shared")
            self.executor.sync_pages(self.pool.pages())
        capacity = dict(self.pool.capacity_bytes_by_component)
        physical_allocations = sum(len(allocator._allocations)
                                   for allocator in self.runtime.allocators.values())
        ownership_ok = (self.read_bytes.get("hbm", 0) == 0
                        and self.append_bytes.get("hbm", 0) == 0
                        and self.read_bytes.get("hbf", 0) > 0
                        and self.append_bytes.get("hbf", 0) > 0)
        checks = {
            "capacity_including_migration_copies": all(self.peak_with_copy[k] <= capacity[k] for k in capacity),
            "all_admitted_sessions_released": self.released_sessions == len(self.admitted),
            "all_logical_and_physical_pages_released": not self.pool.pages() and physical_allocations == 0,
            "all_hbf_active_reads_and_appends": ownership_ok if self.architecture == "B_all_hbf" else True,
            "pause_marks_retained_in_every_architecture": self.peak_retained > 0,
            "resume_observed": bool(self.resume_waits),
            "hot_cold_has_both_migration_directions": (
                {x.kind for x in self.pool.events} >= {"offload", "restore"}
                if self.architecture == "C_hot_cold" else True),
            "prefix_reuse_observed": self.prefix_hits > 0 if reuse_count else True,
            "offered_session_conservation": len(self.admitted) + len(self.rejected) == p["sessions"],
            "prefix_offer_accounting": self.prefix_offers == reuse_count and self.prefix_hits <= self.prefix_offers,
        }
        waits = sorted(self.resume_waits)
        percentile = lambda fraction: waits[min(len(waits) - 1, ceil(len(waits) * fraction) - 1)] if waits else None
        return {
            "architecture": self.architecture, "parameters": p,
            "status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "scope": "physical_memory_session_lifecycle_microbenchmark",
            "llm_token_per_second": None, "ttft_ns": None, "tpot_ns": None,
            "slo_goodput": None,
            "admitted_sessions": len(self.admitted), "rejected_sessions": len(self.rejected),
            "active_peak": self.peak_active, "retained_peak": self.peak_retained,
            "live_peak": self.peak_live, "released_sessions": self.released_sessions,
            "prefix_hits": self.prefix_hits,
            "prefix_offered_sessions": self.prefix_offers,
            "prefix_reuse_rate_offered": self.prefix_offers / p["sessions"],
            "prefix_reuse_rate_admitted": self.prefix_hits / len(self.admitted) if self.admitted else None,
            "prefix_success_fraction_of_offered_sessions": self.prefix_hits / p["sessions"],
            "prefix_reuse_denominators": {
                "requested_rate": "target fraction of all offered sessions, including the initial non-reusing seed session",
                "offered_rate": "prefix_offered_sessions / parameters.sessions; independent of admission",
                "admitted_rate": "prefix_hits / admitted_sessions; capacity rejection can change this fraction",
            },
            "capacity_bytes": capacity, "peak_bytes": self.peak_bytes,
            "peak_bytes_including_migration_copies": self.peak_with_copy,
            "migration_copy_peak_bytes": self.copy_peak,
            "active_read_bytes_by_component": self.read_bytes,
            "active_append_bytes_by_component": self.append_bytes,
            "media_write_bytes_excluding_migration": self.media_write_bytes,
            "migration_bytes": sum(x.bytes for x in self.pool.events),
            "offload_bytes": sum(x.bytes for x in self.pool.events if x.kind == "offload"),
            "restore_bytes": sum(x.bytes for x in self.pool.events if x.kind == "restore"),
            "resume_latency_p50_ns": percentile(.5), "resume_latency_p99_ns": percentile(.99),
            "resume_latencies_ns": self.resume_waits, "observed_pause_durations_ns": self.actual_pause_ns,
            "elapsed_ns": self.clock, "events": self.events,
            "executed_physical_configurations": _executed_core_configurations(self.runtime),
        }


def run_session_experiments() -> dict[str, Any]:
    variants = (
        ("base", {}), ("active_half", {"active_ratio": .5}),
        ("all_active", {"active_ratio": 1.0}),
        ("long_pause", {"pause_ns": 10_000_000.0}),
        ("frequent_resume", {"cycles": 4}),
        ("prefix_0", {"prefix_reuse_rate": 0.0, "sessions": 20}),
        ("prefix_25", {"prefix_reuse_rate": .25, "sessions": 20}),
        ("prefix_50", {"prefix_reuse_rate": .5, "sessions": 20}),
        ("prefix_90", {"prefix_reuse_rate": .9, "sessions": 20}),
        ("hbm_all_fit", {"hbm_pages": 40}),
    )
    rows = []
    for name, parameters in variants:
        for architecture in ("A_hbm_only", "B_all_hbf", "C_hot_cold"):
            row = SessionExperiment(architecture, **parameters).run()
            row["id"] = name + "/" + architecture
            rows.append(row)
    checks = [{"id": row["id"] + "/" + key, "passed": passed,
               "observed": passed, "expected": True}
              for row in rows for key, passed in row["checks"].items()]
    passed = all(c["passed"] for c in checks)
    return {
        "id": "04_hot_cold_sessions", "title": "④ 热冷会话生命周期与前缀共享",
        "status": "passed" if passed else "failed",
        "checks": checks, "rows": rows,
        "capability": {
            "supported": ["真实物理页读取、append、pause/resume迁移与前缀引用共享", "容量受限A/B/C会话数和恢复等待对比"] if passed else [],
            "unsupported": ["完整LLM算子、GPU计算吞吐、TTFT/TPOT/goodput", "商用硬件session/GPU或论文容量倍数复现"],
        },
        "limitations": [
            "Memory/session microbenchmark: no transformer operators, GPU execution or LLM TPS/TTFT/TPOT/goodput claims.",
            "B explicitly assumes memory-accessible HBF and executes active reads and appends there; C migrates private cold pages and keeps shared immutable prefixes hot.",
            "Small synthetic pages and one GPU-equivalent pool; no production session capacity claim or paper result reproduction.",
            "Sequential admission and page transfers; requested lifecycle timestamps are identical for A/B/C but observed pause duration includes queueing and cohort scheduling.",
            "The prefix scan offers 20 sessions at 0/25/50/90% exact requested reuse; offered and admitted denominators are reported separately. The initial prefix-producing session is in the offered denominator and does not itself count as a reuse hit.",
            "No GC, endurance or monetary cost model is enabled in this lifecycle group.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_session_experiments()
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    raise SystemExit(0 if result["status"] == "passed" else 1)
