"""Trace-level HBM/HBF hot/cold session experiments.

This entry point intentionally stays below the serving scheduler.  It uses
the production ``DynamicKVPool`` and ``PhysicalKVTransferExecutor`` to make
pause/resume transfers observable without building a second scheduler model.
Run ``python scripts/hbf_hbm_sessions.py`` for a small deterministic matrix.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.communication import TopologyRouter
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.ir import ComponentSpec, HardwareSpec, LinkSpec, PortSpec
from heterollm_sim.kv_pool import DynamicKVPool, KvPoolComponent
from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
from heterollm_sim.physical_presets import resolve_physical_config


PAGE = 16_384
TOKENS_PER_PAGE = 16


def _executed_core_configurations(runtime: PhysicalRuntimeContext) -> dict[str, Any]:
    """Read parameters from the live Cores that completed physical accesses.

    Binding an allocation can create a runtime without issuing memory work;
    its clock stays zero. Include actual initialization and measured accesses,
    but never resolve a fresh catalog entry as evidence of execution.
    """
    return {owner: {"core": type(active.core).__name__,
                    "physical_config": asdict(active.core.config),
                    "last_completion_ns": active.clock_ns}
            for owner, active in runtime.runtimes.items() if active.clock_ns > 0}


@dataclass(frozen=True)
class SessionEvent:
    sequence: int
    timestamp_ns: float
    event_type: str
    session_id: str
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "timestamp_ns": float(self.timestamp_ns),
            "event_type": self.event_type,
            "session_id": self.session_id,
            "details": dict(self.details),
        }


def _hardware(*, hbm_pages: int, hbf_pages: int) -> HardwareSpec:
    hbm_bytes = int(hbm_pages) * PAGE
    hbf_bytes = int(hbf_pages) * PAGE
    hbm = ComponentSpec(
        "hbm", "hbm", capacity_bytes=hbm_bytes,
        ports=(PortSpec("p0", "HBM", "endpoint", bandwidth_gbps=4096.0),),
        bandwidth_gbps=4096.0,
        metadata={"physical_memory_config": resolve_physical_config(
            "research_hbm_sessions", overrides={"capacity_bytes": hbm_bytes})},
    )
    hbf = ComponentSpec(
        "hbf", "hbf", capacity_bytes=hbf_bytes,
        ports=(PortSpec("p0", "PCIe", "endpoint", bandwidth_gbps=512.0),),
        read_bandwidth_gbps=488.0 * 8.0,
        write_bandwidth_gbps=27.2 * 8.0,
        metadata={"physical_memory_config": resolve_physical_config(
            "research_hbf_sessions", overrides={"capacity_bytes": hbf_bytes})},
    )
    return HardwareSpec(
        "session-hbm-hbf",
        (hbm, hbf),
        (LinkSpec(
            "hbm-hbf", "hbm", "p0", "hbf", "p0", "PCIe",
            bandwidth_gbps=512.0, latency_ns=100.0,
        ),),
        require_connected=False,
    )


class SessionTrace:
    """One deterministic session trace backed by a real physical KV pool."""

    def __init__(
        self,
        *,
        mode: str,
        pause_duration_ns: float,
        resume_cycles: int,
        hbm_pages: int = 6,
        hbf_pages: int = 8,
        hot_pages: int = 1,
    ) -> None:
        if mode not in {"hbm_only", "all_hbf", "hot_cold"}:
            raise ValueError("mode must be hbm_only, all_hbf, or hot_cold")
        if pause_duration_ns < 0 or resume_cycles <= 0:
            raise ValueError("pause duration and resume cycles must be non-negative/positive")
        self.mode = mode
        self.pause_duration_ns = float(pause_duration_ns)
        self.resume_cycles = int(resume_cycles)
        self.hot_pages = max(0, int(hot_pages))
        self.clock = 0.0
        self.events: list[SessionEvent] = []
        self._pool_event_cursor = 0
        self._active: set[str] = set()
        self._retained: set[str] = set()
        self._peak_active = 0
        self._peak_retained = 0
        self._resume_wait_ns: list[float] = []
        self._resume_requests: dict[str, float] = {}
        self._pause_count = 0
        self._end_releases = 0
        hardware = _hardware(hbm_pages=hbm_pages, hbf_pages=hbf_pages)
        self._physical_runtime = PhysicalRuntimeContext()
        self._topology = TopologyRouter(hardware)
        self._executor = PhysicalKVTransferExecutor(
            hardware.components,
            runtime=self._physical_runtime,
            topology=self._topology,
            arrival_ns=lambda _request: self.clock,
        )
        self.pool = DynamicKVPool(
            [
                KvPoolComponent("hbm", hbm_pages * PAGE, kind="hbm", active=True,
                                read_bandwidth_gbps=4096.0, write_bandwidth_gbps=4096.0),
                KvPoolComponent("hbf", hbf_pages * PAGE, kind="hbf", active=False,
                                read_bandwidth_gbps=488.0 * 8.0,
                                write_bandwidth_gbps=27.2 * 8.0),
            ],
            tokens_per_page=TOKENS_PER_PAGE,
            page_bytes=PAGE,
            topology=self._topology,
            offload_components=("hbf",),
            transfer_executor=self._executor,
        )

    def _emit(self, event_type: str, session_id: str, timestamp_ns: float | None = None,
              **details: Any) -> None:
        self.events.append(SessionEvent(
            len(self.events),
            self.clock if timestamp_ns is None else float(timestamp_ns),
            event_type, session_id, details,
        ))

    def _sync_physical_pages(self) -> None:
        self._executor.sync_pages(self.pool.pages())

    def _capture_transfers(self) -> None:
        transfers = self.pool.events
        for transfer in transfers[self._pool_event_cursor:]:
            start = float(transfer.start_ns if transfer.start_ns is not None else self.clock)
            completion = float(
                transfer.completion_ns
                if transfer.completion_ns is not None
                else start + transfer.duration_ns
            )
            details = {
                "kind": transfer.kind,
                "logical_page_id": int(transfer.logical_page_id),
                "source_component": transfer.source_component,
                "target_component": transfer.target_component,
                "bytes": int(transfer.bytes),
                "duration_ns": float(transfer.duration_ns),
            }
            page = self.pool.page(transfer.logical_page_id)
            session_id = page.request_id if page is not None else ""
            self.events.append(SessionEvent(
                len(self.events), start, "migration_start", session_id, details
            ))
            self.events.append(SessionEvent(
                len(self.events), completion, "migration_complete", session_id, details
            ))
            self.clock = max(self.clock, completion)
        self._pool_event_cursor = len(transfers)

    def _pages_for(self, session_id: str) -> tuple[Any, ...]:
        return self.pool.request_pages(session_id)

    def _create(self, session_id: str, pages: int) -> None:
        self._emit("create", session_id, pages=pages)
        if not self.pool.resize_detailed(session_id, pages).success:
            raise RuntimeError(self.pool.last_error or "session allocation failed")
        self._sync_physical_pages()
        self._active.add(session_id)
        self._peak_active = max(self._peak_active, len(self._active))

    def _append(self, session_id: str, pages: int = 1) -> None:
        before = len(self._pages_for(session_id))
        target = before + int(pages)
        self._emit("append", session_id, pages=pages, target_pages=target)
        if not self.pool.resize_detailed(session_id, target).success:
            raise RuntimeError(self.pool.last_error or "session append failed")
        self._sync_physical_pages()

    def _pause(self, session_id: str) -> None:
        if session_id not in self._active:
            raise RuntimeError("only active sessions can pause")
        pause_start = self.clock
        self._emit("pause", session_id, pause_start)
        self._active.remove(session_id)
        self._retained.add(session_id)
        self._peak_retained = max(self._peak_retained, len(self._retained))
        pages = list(self._pages_for(session_id))
        if self.mode == "all_hbf":
            selected = pages
        elif self.mode == "hot_cold":
            selected = pages[self.hot_pages:]
        else:
            selected = []
        moved = 0
        for page in selected:
            if page.owner_component == "hbm":
                if not self.pool.offload_page(page.logical_page_id, "hbf"):
                    raise RuntimeError(self.pool.last_error or "session offload failed")
                moved += 1
        self._capture_transfers()
        self._pause_count += 1
        self._emit("pause_complete", session_id, self.clock,
                   pages_migrated=moved, pause_duration_ns=self.clock - pause_start)
        resume_request = pause_start + self.pause_duration_ns
        self._emit("resume_requested", session_id, resume_request)
        self._resume_requests[session_id] = resume_request
        self.clock = max(self.clock, resume_request)

    def _resume(self, session_id: str) -> None:
        if session_id not in self._retained:
            raise RuntimeError("only retained sessions can resume")
        requested = self._resume_requests.pop(session_id, self.clock)
        self.clock = max(self.clock, requested)
        pages = self._pages_for(session_id)
        restored = 0
        if self.mode in {"all_hbf", "hot_cold"}:
            for page in pages:
                if page.owner_component == "hbf":
                    if not self.pool.restore_page(page.logical_page_id, "hbm", kind="restore"):
                        raise RuntimeError(self.pool.last_error or "session restore failed")
                    restored += 1
        self._capture_transfers()
        actual = self.clock
        self._resume_wait_ns.append(max(0.0, actual - requested))
        self._retained.remove(session_id)
        self._active.add(session_id)
        self._peak_active = max(self._peak_active, len(self._active))
        self._emit("resume", session_id, actual,
                   pages_restored=restored,
                   requested_ns=requested,
                   restore_wait_ns=max(0.0, actual - requested))

    def _end(self, session_id: str) -> None:
        if session_id not in self._active and session_id not in self._retained:
            raise RuntimeError("only live sessions can end")
        pages_before = len(self._pages_for(session_id))
        self._emit("end", session_id, pages=pages_before)
        self.pool.release(session_id)
        self._sync_physical_pages()
        self._active.discard(session_id)
        self._retained.discard(session_id)
        self._end_releases += pages_before

    def run(self) -> dict[str, Any]:
        self._create("session-hot", 2)
        self._create("session-cold", 2)
        self._append("session-hot")
        self._pause("session-cold")
        self._resume("session-cold")
        for _ in range(self.resume_cycles - 1):
            self._pause("session-cold")
            self._resume("session-cold")
        self._pause("session-hot")
        self._resume("session-hot")
        self._end("session-cold")
        self._end("session-hot")
        remaining = len(self.pool.pages())
        if remaining or any(value for value in self.pool.used_bytes_by_component.values()):
            raise AssertionError("ending all sessions must release every KV page")
        migration_events = [event for event in self.events if event.event_type == "migration_complete"]
        offload_events = [
            event for event in migration_events
            if event.details.get("kind") == "offload"
        ]
        restore_events = [
            event for event in migration_events
            if event.details.get("kind") == "restore"
        ]
        migration_bytes = sum(int(event.details["bytes"]) for event in migration_events)
        restore_wait = max(self._resume_wait_ns, default=0.0)
        memory_phases = [
            phase for event in self._executor.events for phase in event["phases"]
            if phase.get("kind") in {"source_read", "target_write", "relay_read", "relay_write"}
        ]
        return {
            "mode": self.mode,
            "placement_semantics": (
                "all paused pages offload to HBF and restore to HBM before execution; not active all-KV-HBF"
                if self.mode == "all_hbf" else
                "first hot_pages pages stay in HBM; other paused pages offload and restore before execution"
                if self.mode == "hot_cold" else "all pages remain in HBM"
            ),
            "scope": "allocation/migration lifecycle only; append allocates pages and does not model token computation or KV writes",
            "pause_duration_ns": self.pause_duration_ns,
            "resume_cycles": self.resume_cycles,
            "events": [event.as_dict() for event in sorted(
                self.events,
                key=lambda event: (event.timestamp_ns, event.sequence),
            )],
            "event_types": sorted({event.event_type for event in self.events}),
            "active_sessions_peak": self._peak_active,
            "retained_sessions_peak": self._peak_retained,
            "pause_count": self._pause_count,
            "migration_count": len(migration_events),
            "migration_bytes": migration_bytes,
            "offload_count": len(offload_events),
            "offload_bytes": sum(int(event.details["bytes"]) for event in offload_events),
            "restore_count": len(restore_events),
            "restore_bytes": sum(int(event.details["bytes"]) for event in restore_events),
            "restore_wait_ns": restore_wait,
            "released_pages": self._end_releases,
            "remaining_pages": remaining,
            "used_bytes_by_component": dict(self.pool.used_bytes_by_component),
            "capacity_bytes_by_component": dict(self.pool.capacity_bytes_by_component),
            "physical_transfer_count": self._executor.counters["transfer_count"],
            "logical_transfer_read_bytes": self._executor.counters["read_bytes"],
            "logical_transfer_write_bytes": self._executor.counters["write_bytes"],
            "physical_read_bytes": sum(int(phase.get("physical_read_bytes", 0)) for phase in memory_phases),
            "physical_write_bytes": sum(int(phase.get("physical_write_bytes", 0)) for phase in memory_phases),
            "makespan_ns": self.clock,
            "executed_physical_configurations": _executed_core_configurations(self._physical_runtime),
        }


def run_experiment(mode: str, pause_duration_ns: float, resume_cycles: int) -> dict[str, Any]:
    return SessionTrace(
        mode=mode,
        pause_duration_ns=pause_duration_ns,
        resume_cycles=resume_cycles,
    ).run()


def run_matrix() -> dict[str, Any]:
    rows = [
        run_experiment(mode, pause, cycles)
        for mode in ("hbm_only", "all_hbf", "hot_cold")
        for pause in (0.0, 250_000.0)
        for cycles in (1, 2)
    ]
    return {
        "schema_version": 1,
        "scope": "two-session physical KV pause/resume trace",
        "scan_axes": {
            "mode": ["hbm_only", "all_hbf", "hot_cold"],
            "pause_duration_ns": [0.0, 250_000.0],
            "resume_cycles": [1, 2],
        },
        "experiments": rows,
        "limitations": [
            "session lifecycle is a focused trace harness below the serving scheduler",
            "all_hbf is the legacy name for full paused-session offload, not decode directly from HBF",
            "two sessions are scripted; active/retained peaks do not measure maximum supported session capacity",
            "hot_cold keeps the first page resident and offloads later pages",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_matrix()
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
