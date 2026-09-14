"""Extract CUDA activity and conservative NVTX operator evidence from Nsight SQLite."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heterollm_sim.kernel_mapping import classify_kernel


class TraceSchemaError(ValueError):
    pass


_OPERATOR_LABEL = re.compile(
    r"^\s*(?:operator|op)\s*(?::|=|/)\s*(?P<name>.+?)\s*$|^\s*\[\s*(?:operator|op)\s*\]\s*(?P<bracket_name>.+?)\s*$",
    re.IGNORECASE,
)
_PHASE_LABEL = re.compile(r"^\s*phase\s*(?::|=|/)\s*(?P<name>.+?)\s*$", re.IGNORECASE)
_REQUEST_MARKER_LABEL = re.compile(
    r"^\s*(?P<name>request_begin|prefill_begin|prefill_end|first_token|request_end|engine_request_begin|engine_request_end|engine_token_begin|engine_token_end|engine_compute_begin|engine_compute_end)(?:\||$)",
    re.IGNORECASE,
)


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f'pragma table_info("{table}")')}


def _column_names(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f'pragma table_info("{table}")')]


def _rows(conn: sqlite3.Connection, table: str) -> list[dict]:
    columns = _column_names(conn, table)
    if not columns:
        return []
    order = " order by start" if "start" in columns else ""
    return [dict(zip(columns, row)) for row in conn.execute(f' select * from "{table}"{order}')]


def _selected(row: dict, start_ns: int | None, end_ns: int | None) -> bool:
    begin = row.get("start", row.get("start_ns"))
    finish = row.get("end", row.get("end_ns"))
    if finish is None:
        finish = begin
    return not ((start_ns is not None and finish < start_ns) or (end_ns is not None and begin > end_ns))


def _string(strings: dict[int, str], value: object) -> str | None:
    if value is None:
        return None
    return strings.get(value, str(value))


def _first(row: dict, *names: str) -> object:
    return next((row[name] for name in names if row.get(name) is not None), None)


def _nvtx_label(row: dict, strings: dict[int, str]) -> str | None:
    label = _first(row, "text", "message", "name", "label")
    if label is not None:
        return str(label)
    return _string(strings, _first(row, "textId", "messageId", "nameId", "labelId"))


def _explicit_operator(label: str | None) -> str | None:
    match = _OPERATOR_LABEL.match(label or "")
    return (match.group("name") or match.group("bracket_name")).strip() if match else None


def _operator_id(label: str | None) -> str | None:
    """Return the stable operator invocation id encoded before ``|`` fields.

    Semantic scopes emitted by the instrumented backend use
    ``operator:<ggml-op>:<tensor-name>|shape=...|type=...``.  The tensor name
    carries the graph/layer invocation identity; shape and dtype are metadata,
    not part of the id.  Keeping this separate from ``semantic_operator``
    preserves the historical label while giving downstream calibration a
    collision-resistant key without any timing-based inference.
    """
    name = _explicit_operator(label)
    if name is None:
        return None
    return name.split("|", 1)[0].strip() or None


def _explicit_phase(label: str | None) -> str | None:
    match = _PHASE_LABEL.match(label or "")
    return match.group("name").strip() if match else None


def _explicit_request_marker(label: str | None) -> str | None:
    """Return the lifecycle marker prefix emitted by semantic llama.cpp."""
    match = _REQUEST_MARKER_LABEL.match(label or "")
    return match.group("name").lower() if match else None


def _label_field(label: str | None, field: str) -> str | None:
    match = re.search(r"(?:^|[|;])" + re.escape(field) + r"=([^|;]+)", label or "", re.IGNORECASE)
    return match.group(1).strip() if match else None


def _semantic_unknown(event: dict, reason: str) -> None:
    event.update({
        "semantic_operator": None,
        "semantic_status": "unknown",
        "semantic_source": None,
        "semantic_reason": reason,
    })


def _attach_operator(event: dict, runtime_by_correlation: dict, nvtx_events: list[dict]) -> None:
    correlation_id = event.get("correlation_id")
    runtimes = runtime_by_correlation.get(correlation_id, []) if correlation_id is not None else []
    if len(runtimes) != 1:
        _semantic_unknown(event, "graph_activity_without_runtime_nvtx_correlation" if event.get("graph_node_id") is not None else "no_unique_runtime_correlation")
        return
    runtime = runtimes[0]
    thread_id = runtime.get("global_tid")
    if thread_id is None:
        _semantic_unknown(event, "runtime_thread_missing")
        return
    scopes = [scope for scope in nvtx_events if scope["operator_name"] is not None and scope["global_tid"] == thread_id and scope["start_ns"] <= runtime["start_ns"] and runtime["end_ns"] <= scope["end_ns"]]
    if not scopes:
        _semantic_unknown(event, "no_containing_explicit_operator_nvtx_range")
        return
    scope = min(scopes, key=lambda item: (item["duration_ns"], -item["start_ns"]))
    event.update({
        "semantic_operator": scope["operator_name"],
        "semantic_operator_id": scope.get("operator_id") or _operator_id(scope.get("label")),
        "semantic_status": "matched",
        "semantic_source": "NVTX_EVENTS/runtime_correlation_same_thread_containment",
        "semantic_reason": "innermost_explicit_operator_nvtx_range",
        "nvtx_event_id": scope["nvtx_event_id"],
        "runtime_global_tid": thread_id,
        "runtime_start_ns": runtime["start_ns"],
        "runtime_end_ns": runtime["end_ns"],
        "semantic_shape": _label_field(scope.get("label"), "shape"),
        "semantic_type": _label_field(scope.get("label"), "type"),
        "semantic_layout": _label_field(scope.get("label"), "layout"),
        "nvtx": {"source": scope["source"], "label": scope["label"], "start_ns": scope["start_ns"], "end_ns": scope["end_ns"]},
    })


def extract_host_boundary_markers(path: Path) -> dict[str, object]:
    """Extract instantaneous B-10 server boundary markers from NVTX_EVENTS."""
    import sqlite3
    labels = {"http.handler.begin", "http.handler.end", "json.tokenize", "slot.launch", "response.queue", "sink.write"}
    conn = sqlite3.connect(str(path))
    try:
        cols = {row[1] for row in conn.execute("pragma table_info( NVTX_EVENTS )")}
        if "text" not in cols or "start" not in cols:
            return {"status": "missing_nvtx_events", "events": [], "counts": {}}
        rows = conn.execute("select start, text from NVTX_EVENTS where text is not null order by start").fetchall()
    finally:
        conn.close()
    events = [{"label": str(text), "start_ns": int(start), "event_kind": "instant"}
              for start, text in rows if str(text) in labels]
    counts = {label: sum(1 for event in events if event["label"] == label) for label in sorted(labels)}
    return {"status": "captured", "events": events, "counts": counts,
            "labels": sorted(labels), "source": str(path)}


def _engine_marker_summary(markers: list[dict]) -> dict[str, object]:
    """Summarize engine boundary markers without inferring a cost model.

    Token operator spans and the gaps between them are deliberately reported
    separately.  A gap may contain scheduling, synchronization, sampling, or
    profiling overhead; treating it as kernel time would double count or
    manufacture an operator cost.
    """
    engine = [item for item in markers
              if item.get("engine_boundary") and item.get("is_instant")]
    begins = [item for item in engine if item.get("engine_boundary") == "engine_request_begin"]
    ends = [item for item in engine if item.get("engine_boundary") == "engine_request_end"]
    token_begins = {}
    token_ends = {}
    for item in engine:
        index = item.get("engine_token_index")
        if index is None:
            continue
        if item.get("engine_boundary") == "engine_token_begin":
            token_begins.setdefault(str(index), item)
        elif item.get("engine_boundary") == "engine_token_end":
            token_ends.setdefault(str(index), item)
    token_indices = sorted(set(token_begins) & set(token_ends), key=lambda value: int(value))
    token_intervals_ms = [
        (float(token_ends[index]["start_ns"]) - float(token_begins[index]["start_ns"])) / 1e6
        for index in token_indices
    ]
    inter_token_gaps_ms = [
        (float(token_begins[next_index]["start_ns"]) - float(token_ends[index]["start_ns"])) / 1e6
        for index, next_index in zip(token_indices, token_indices[1:])
    ]
    inter_token_begin_gaps_ms = [
        (float(token_begins[next_index]["start_ns"]) - float(token_begins[index]["start_ns"])) / 1e6
        for index, next_index in zip(token_indices, token_indices[1:])
    ]
    request_interval_ms = None
    if begins and ends:
        request_interval_ms = (float(max(ends, key=lambda item: item["start_ns"])["start_ns"]) -
                                float(min(begins, key=lambda item: item["start_ns"])["start_ns"])) / 1e6
    return {
        "status": "complete" if len(begins) == 1 and len(ends) == 1 and
                  len(token_indices) == len(token_begins) == len(token_ends) and
                  all(value >= 0.0 for value in inter_token_gaps_ms) else "incomplete",
        "request_begin_count": len(begins),
        "request_end_count": len(ends),
        "token_indices": token_indices,
        "token_begin_count": len(token_begins),
        "token_end_count": len(token_ends),
        "token_intervals_ms": token_intervals_ms,
        "inter_token_gaps_ms": inter_token_gaps_ms,
        "inter_token_begin_gaps_ms": inter_token_begin_gaps_ms,
        "engine_request_interval_ms": request_interval_ms,
        "cost_inference": "forbidden_without_boundary_and_nonprofiling_validation",
    }

def extract(path: Path, start_ns: int | None = None, end_ns: int | None = None):
    conn = sqlite3.connect(str(path))
    try:
        tables = {row[0] for row in conn.execute("select name from sqlite_master where type='table'")}
        required = {"StringIds", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_RUNTIME"}
        missing = sorted(required - tables)
        if missing:
            raise TraceSchemaError("Nsight SQLite schema missing tables: " + ", ".join(missing))
        for table, required_columns in (("CUPTI_ACTIVITY_KIND_KERNEL", ("start", "end", "demangledName")), ("CUPTI_ACTIVITY_KIND_MEMCPY", ("start", "end", "bytes", "copyKind")), ("CUPTI_ACTIVITY_KIND_RUNTIME", ("start", "end", "nameId"))):
            absent = sorted(set(required_columns) - _columns(conn, table))
            if absent:
                raise TraceSchemaError(f"Nsight SQLite table {table} missing columns: {', '.join(absent)}")
        strings = {int(key): value for key, value in conn.execute("select id,value from StringIds")}
        runtime_rows = _rows(conn, "CUPTI_ACTIVITY_KIND_RUNTIME")
        runtime_by_correlation: dict[object, list[dict]] = {}
        events = []
        for row in runtime_rows:
            runtime = {
                "kind": "runtime", "start_ns": row["start"], "end_ns": row["end"], "duration_ns": row["end"] - row["start"],
                "correlation_id": row.get("correlationId"), "global_tid": _first(row, "globalTid", "threadId"),
                "runtime_name": _string(strings, row.get("nameId")),
            }
            if runtime["correlation_id"] is not None:
                runtime_by_correlation.setdefault(runtime["correlation_id"], []).append(runtime)
            if _selected(row, start_ns, end_ns):
                events.append(runtime)
        nvtx_events = []
        if "NVTX_EVENTS" in tables:
            nvtx_columns = _columns(conn, "NVTX_EVENTS")
            absent = sorted({"start", "end"} - nvtx_columns)
            if absent:
                raise TraceSchemaError("Nsight SQLite table NVTX_EVENTS missing columns: " + ", ".join(absent))
            for index, row in enumerate(_rows(conn, "NVTX_EVENTS")):
                label = _nvtx_label(row, strings)
                # NVTX marks (eventType 34) are points and have NULL end in
                # Nsight SQLite.  Keep them distinct from asynchronous ranges
                # carrying the same lifecycle label so boundaries count once.
                finish = row["end"] if row["end"] is not None else row["start"]
                nvtx_events.append({
                    "nvtx_event_id": index, "kind": "nvtx", "source": "NVTX_EVENTS", "start_ns": row["start"], "end_ns": finish, "duration_ns": finish - row["start"],
                    "nvtx_event_type": row.get("eventType"),
                    "is_instant": row.get("eventType") == 34 or (row.get("eventType") is None and finish == row["start"]),
                    "global_tid": _first(row, "globalTid", "threadId"), "label": label,
                    "operator_name": _explicit_operator(label), "operator_id": _operator_id(label),
                    "phase_name": _explicit_phase(label),
                    "request_marker": _explicit_request_marker(label),
                    "engine_boundary": (_explicit_request_marker(label)
                                         if (_explicit_request_marker(label) or "").startswith("engine_")
                                         else None),
                    "engine_token_index": _label_field(label, "token_index"),
                    "engine_token_count": _label_field(label, "token_count"),
                    "raw": row,
                })
        for table, kind in (("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"), ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy")):
            for row in _rows(conn, table):
                if not _selected(row, start_ns, end_ns):
                    continue
                event = {"kind": kind, "start_ns": row["start"], "end_ns": row["end"], "duration_ns": row["end"] - row["start"], "device_id": row.get("deviceId"), "stream_id": row.get("streamId"), "correlation_id": row.get("correlationId"), "graph_node_id": row.get("graphNodeId")}
                if kind == "kernel":
                    kernel_name = _string(strings, row.get("demangledName"))
                    classification = classify_kernel(kernel_name or "", row)
                    stage_evidence = "explicit_annotation" if classification.reason.startswith("explicit ") else ("name_heuristic_only" if classification.confidence else "unknown")
                    event.update({"kernel_name": kernel_name, "short_name": _string(strings, row.get("shortName")), "stage": classification.stage, "stage_confidence": classification.confidence, "stage_reason": classification.reason, "stage_evidence": stage_evidence, "grid": [row.get("gridX"), row.get("gridY"), row.get("gridZ")], "block": [row.get("blockX"), row.get("blockY"), row.get("blockZ")]})
                else:
                    event.update({"bytes": row.get("bytes"), "copy_kind": row.get("copyKind"), "src_kind": row.get("srcKind"), "dst_kind": row.get("dstKind")})
                _attach_operator(event, runtime_by_correlation, nvtx_events)
                # Operator ownership is attached from NVTX/runtime correlation,
                # so perform the conservative semantic stage pass afterwards.
                # No GPU-time overlap is used here.  Existing explicit/name
                # stages remain authoritative; only ``unknown`` is refined.
                if kind == "kernel" and event.get("stage") == "unknown" and event.get("semantic_status") == "matched":
                    semantic_classification = classify_kernel(kernel_name or "", event)
                    if semantic_classification.stage != "unknown":
                        event.update({
                            "stage": semantic_classification.stage,
                            "stage_confidence": semantic_classification.confidence,
                            "stage_reason": semantic_classification.reason,
                            "stage_evidence": "semantic_operator_owner",
                        })
                events.append(event)
        graph_nodes = []
        for row in _rows(conn, "CUDA_GRAPH_NODE_EVENTS"):
            if _selected(row, start_ns, end_ns):
                graph_nodes.append({"kind": "graph_node", "metadata_only": True, "start_ns": row["start"], "end_ns": row["end"], "graph_node_id": row.get("graphNodeId"), "original_graph_node_id": row.get("originalGraphNodeId"), "event_name": _string(strings, row.get("nameId"))})
    finally:
        conn.close()
    events.sort(key=lambda event: (event["start_ns"], event["end_ns"], event["kind"]))
    visible_nvtx = [event for event in nvtx_events if _selected(event, start_ns, end_ns)]
    request_markers = [event for event in visible_nvtx if event.get("request_marker") is not None]
    return {
        "schema": "native-nsys-trace/v1", "source": str(path.resolve()),
        "event_count": len(events), "events": events,
        "nvtx_event_count": len(visible_nvtx), "nvtx_events": visible_nvtx,
        "request_marker_count": len(request_markers),
        "request_markers": request_markers,
        "engine_marker_summary": _engine_marker_summary(request_markers),
        "graph_node_metadata_count": len(graph_nodes), "graph_node_metadata": graph_nodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-ns", type=int)
    parser.add_argument("--end-ns", type=int)
    args = parser.parse_args()
    output = extract(args.sqlite, args.start_ns, args.end_ns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"schema": output["schema"], "event_count": output["event_count"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
