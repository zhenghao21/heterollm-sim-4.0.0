from __future__ import annotations

import json

from tools.extract_cpu_perf_trace import build


def test_extract_cpu_perf_selects_formal_graphs_and_phase(tmp_path):
    log = tmp_path / "cpu.log"
    rows = []
    for graph, m in ((1, 2), (2, 1), (3, 2), (4, 1)):
        rows.extend([
            f"CPU_PERF|op=GET_ROWS|name=model.input_embed|shape=[8,{m},1,1]|type=f32|src0_type=q8_0|src0_shape=[8,32,1,1]|src1_type=i32|src1_shape=[{m},1,1,1]|src0_bytes=64|src1_bytes=8|dst_bytes=32|total_bytes=104|threads=2|graph_id={graph}|node_n=0|fused=0|start_us={graph*100}|end_us={graph*100+2}|duration_us=2",
            f"CPU_PERF|op=MUL_MAT|name=ffn_gate-{graph}|shape=[16,{m},1,1]|type=f32|src0_type=q4_k|src0_shape=[8,16,1,1]|src1_type=f32|src1_shape=[8,{m},1,1]|src0_bytes=128|src1_bytes=64|dst_bytes=128|total_bytes=320|threads=2|graph_id={graph}|node_n=1|fused=0|start_us={graph*100+3}|end_us={graph*100+8}|duration_us=5",
        ])
    log.write_text("\n".join(rows), encoding="utf-8")
    trace = build(log, formal_graphs=2)
    assert trace["cpu_operator_trace"]["selected_invocation_indices"] == [2, 3]
    assert len(trace["events"]) == 4
    assert [event["phase"] for event in trace["events"]] == ["prefill", "prefill", "decode", "decode"]
    assert trace["events"][1]["semantic_operator_id"] == "MUL_MAT:ffn_gate-3"
    assert trace["events"][1]["total_bytes"] == 320
