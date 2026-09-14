"""Convert opt-in llama.cpp CPU_PERF lines into a semantic trace.

The CPU backend emits one line per graph node when
``GGML_CPU_OPERATOR_TRACE=1``.  A native request consists of a warmup graph
sequence followed by the formal sequence; by default the last three graph
invocations (prompt plus two decode steps for predict=2) are retained.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


_LIST_RE = re.compile(r"^\[(.*)\]$")


def _value(text: str):
    text = text.strip()
    match = _LIST_RE.match(text)
    if match:
        try:
            return ast.literal_eval("[" + match.group(1) + "]")
        except (SyntaxError, ValueError):
            return text
    return text


def _parse_line(line: str) -> dict | None:
    if not line.startswith("CPU_PERF|"):
        return None
    values: dict[str, object] = {}
    for field in line.rstrip().split("|")[1:]:
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        values[key] = _value(value)
    if not values.get("op") or values.get("start_us") is None:
        return None
    for key in ("start_us", "end_us", "duration_us", "threads", "graph_id", "node_n", "fused",
                "src0_bytes", "src1_bytes", "dst_bytes", "total_bytes"):
        try:
            values[key] = int(float(values[key]))
        except (KeyError, TypeError, ValueError):
            values[key] = 0
    return values


def _shape(value: object) -> str | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return "x".join(str(int(item)) for item in value)
        except (TypeError, ValueError):
            return None
    return None


def _phase(shape: object) -> str | None:
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        try:
            # ggml tensor ne[1] is the token M for the graph node output.
            return "prefill" if int(shape[1]) > 1 else "decode"
        except (TypeError, ValueError):
            return None
    return None


def build(log_path: Path, *, formal_graphs: int = 3) -> dict:
    records = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        item = _parse_line(line)
        if item is not None:
            records.append(item)
    # The model input embedding is the first node in each graph invocation.
    starts = [i for i, item in enumerate(records)
              if item.get("op") == "GET_ROWS" and item.get("name") == "model.input_embed"]
    if not starts:
        raise ValueError("CPU_PERF log has no model.input_embed graph boundary")
    groups = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(records)
        groups.append(records[start:end])
    if formal_graphs <= 0:
        raise ValueError("formal_graphs must be positive")
    selected = groups[-formal_graphs:]
    events = []
    invocation_offset = len(groups) - len(selected)
    for invocation_index, group in enumerate(selected, invocation_offset):
        for item in group:
            out_shape = item.get("shape")
            out_shape_key = _shape(out_shape)
            operator_id = f"{item['op']}:{item.get('name', '')}"
            events.append({
                "kind": "cpu_operator",
                "backend": "cpu",
                "duration_ns": int(item.get("duration_us", 0)) * 1000,
                "start_ns": int(item.get("start_us", 0)) * 1000,
                "end_ns": int(item.get("end_us", 0)) * 1000,
                "semantic_status": "matched",
                "semantic_source": "GGML_CPU_OPERATOR_TRACE",
                "semantic_operator": operator_id,
                "semantic_operator_id": operator_id,
                "operator_id": operator_id,
                "operator_name": item.get("name", ""),
                "stage_evidence": "explicit_cpu_operator_trace",
                "phase": _phase(out_shape),
                "token_shape": out_shape_key,
                "shape": out_shape_key,
                "type": item.get("type"),
                "src0_type": item.get("src0_type"),
                "src0_shape": _shape(item.get("src0_shape")),
                "src1_type": item.get("src1_type"),
                "src1_shape": _shape(item.get("src1_shape")),
                "src0_bytes": item.get("src0_bytes", 0),
                "src1_bytes": item.get("src1_bytes", 0),
                "dst_bytes": item.get("dst_bytes", 0),
                "total_bytes": item.get("total_bytes", 0),
                "threads": item.get("threads", 0),
                "graph_id": item.get("graph_id", 0),
                "invocation_index": invocation_index,
                "node_n": item.get("node_n", 0),
                "fused": item.get("fused", 0),
            })
    return {
        "schema": "native-nsys-trace/v1",
        "source": str(log_path.resolve()),
        "backend": "cpu",
        "event_count": len(events),
        "events": events,
        "nvtx_events": [],
        "cpu_operator_trace": {
            "enabled": True,
            "formal_graphs_requested": formal_graphs,
            "graph_groups_total": len(groups),
            "selected_invocation_indices": list(range(invocation_offset, len(groups))),
            "phase_evidence": "output tensor ne[1] (M); M>1 prefill, M=1 decode",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal-graphs", type=int, default=3)
    args = parser.parse_args()
    result = build(args.log, formal_graphs=args.formal_graphs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"schema": result["schema"], "event_count": result["event_count"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
