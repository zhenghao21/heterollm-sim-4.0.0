"""Attribute CUDA runtime API calls to explicit llama.cpp phase ranges.

Only calls fully contained in one same-thread ``phase:prefill`` or
``phase:decode`` NVTX range are reported.  Calls crossing a boundary are
discarded instead of guessed, which keeps launch/synchronization evidence
separate from operator kernel coefficients.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict, Counter
from pathlib import Path

TARGETS = {
    "cudaLaunchKernel_v7000": "launch",
    "cudaLaunchKernelExC_v11060": "launch",
    "cudaStreamSynchronize_v3020": "synchronize",
}


def extract(path: Path) -> dict:
    conn = sqlite3.connect(path)
    try:
        strings = dict(conn.execute("SELECT id, value FROM StringIds"))
        nvtx_columns = {row[1] for row in conn.execute("pragma table_info('NVTX_EVENTS')")}
        runtime_columns = {row[1] for row in conn.execute("pragma table_info('CUPTI_ACTIVITY_KIND_RUNTIME')")}
        required_runtime = {"start", "end", "nameId"}
        missing_runtime = sorted(required_runtime - runtime_columns)
        if missing_runtime:
            raise ValueError("CUPTI_ACTIVITY_KIND_RUNTIME missing columns: " + ", ".join(missing_runtime))
        if "start" not in nvtx_columns or "end" not in nvtx_columns:
            raise ValueError("NVTX_EVENTS missing start/end columns")
        # Nsight exports may put the label in textId rather than text.  Resolve
        # both forms so a valid phase range is not silently lost.
        nvtx_rows = [dict(zip(
            [row[1] for row in conn.execute("pragma table_info('NVTX_EVENTS')")], values
        )) for values in conn.execute("select * from NVTX_EVENTS")]
        phases = []
        instant_phase_markers = 0
        phase_missing_label = 0
        for row in nvtx_rows:
            text = row.get("text")
            if text is None and row.get("textId") is not None:
                text = strings.get(row["textId"])
            label = str(text or "").strip()
            if not label.lower().startswith("phase:"):
                continue
            if row.get("end") is None:
                # Instant NVTX marks are lifecycle points, not phase ranges.
                instant_phase_markers += 1
                continue
            if ":" not in label:
                phase_missing_label += 1
                continue
            tid = row.get("globalTid")
            if tid is None:
                phase_missing_label += 1
                continue
            phases.append((int(row["start"]), int(row["end"]), label.split(":", 1)[1].strip(), int(tid)))
        totals: dict[str, dict[str, dict[str, float | int]]] = defaultdict(
            lambda: defaultdict(lambda: {"count": 0, "total_ns": 0})
        )
        excluded = 0
        excluded_reasons: Counter[str] = Counter()
        runtime_names = ["start", "end", "eventClass", "globalTid", "correlationId", "nameId", "returnValue", "callchainId"]
        selected_runtime = [name for name in runtime_names if name in runtime_columns]
        for values in conn.execute("select " + ",".join(selected_runtime) + " from CUPTI_ACTIVITY_KIND_RUNTIME"):
            row = dict(zip(selected_runtime, values))
            start, end = row.get("start"), row.get("end")
            if start is None or end is None:
                excluded_reasons["runtime_missing_boundary"] += 1
                continue
            name_id = row.get("nameId")
            name = strings.get(name_id, str(name_id))
            kind = TARGETS.get(name)
            if kind is None:
                continue
            tid = row.get("globalTid")
            matching = [
                phase for ps, pe, phase, ptid in phases
                if ptid == tid and ps <= start and end <= pe
            ]
            if len(matching) != 1:
                excluded += 1
                if tid is None:
                    excluded_reasons["missing_runtime_thread"] += 1
                elif not matching:
                    excluded_reasons["no_unique_phase_containment"] += 1
                else:
                    excluded_reasons["ambiguous_phase_containment"] += 1
                continue
            bucket = totals[matching[0]][kind]
            bucket["count"] += 1
            bucket["total_ns"] += int(end - start)
        return {
            "schema": "native-cuda-api-phase/v1",
            "source": str(path.resolve()),
            "phase_ranges": {
                phase: sum(1 for _ps, _pe, item, _tid in phases if item == phase)
                for phase in ("prefill", "decode")
            },
            "api": {phase: dict(values) for phase, values in totals.items()},
            "excluded_boundary_calls": excluded,
            "excluded_by_reason": dict(excluded_reasons),
            "instant_phase_markers": instant_phase_markers,
            "phase_missing_label": phase_missing_label,
            "policy": "same-thread API interval fully contained in one explicit NVTX phase; boundary calls excluded; textId labels resolved and instant phase marks ignored",
        }
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.sqlite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"schema": result["schema"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
