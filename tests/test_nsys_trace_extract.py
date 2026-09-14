import sqlite3
from pathlib import Path

import pytest

from tools.extract_nsys_trace import _explicit_request_marker, _operator_id, extract, extract_host_boundary_markers


def _db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript("""
        create table StringIds(id integer primary key,value text);
        create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,deviceId integer,streamId integer,correlationId integer,demangledName integer,shortName integer,graphNodeId integer,gridX integer,gridY integer,gridZ integer,blockX integer,blockY integer,blockZ integer);
        create table CUPTI_ACTIVITY_KIND_MEMCPY(start integer,end integer,deviceId integer,streamId integer,correlationId integer,bytes integer,copyKind integer,srcKind integer,dstKind integer,graphNodeId integer);
        create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,correlationId integer,nameId integer);
        create table NVTX_EVENTS(start integer,end integer,globalTid integer,text text);
    """)
    return connection


def test_extracts_named_kernel_memcpy_runtime(tmp_path: Path):
    db = tmp_path / "x.sqlite"
    connection = _db(db)
    connection.execute("insert into StringIds values(1,'kernel_x')")
    connection.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values(10,30,0,2,7,1,1,9,1,1,1,32,1,1)")
    connection.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values(35,45,0,2,8,128,1,1,2,null)")
    connection.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values(5,8,3,6,1)")
    connection.commit()
    connection.close()

    output = extract(db)

    assert output["event_count"] == 3
    assert output["events"][1]["kernel_name"] == "kernel_x"
    assert output["events"][1]["semantic_status"] == "unknown"


def test_matches_innermost_explicit_operator_only_via_runtime_correlation(tmp_path: Path):
    db = tmp_path / "semantic.sqlite"
    connection = _db(db)
    connection.executemany("insert into StringIds values(?, ?)", [(1, "q_proj_kernel"), (2, "cudaLaunchKernel")])
    connection.executemany("insert into NVTX_EVENTS values(?, ?, ?, ?)", [(0, 100, 7, "phase:prefill"), (0, 100, 7, "operator: outer"), (10, 90, 7, "operator: attention.qkv|shape=128x4x1x1|type=f32|layout=contiguous"), (10, 90, 8, "operator: other-thread"), (12, 20, 7, "helper scope")])
    connection.executemany("insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?, ?, ?, ?, ?)", [(15, 16, 7, 41, 2), (25, 26, 7, 42, 2)])
    connection.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values(30,40,0,2,41,1,1,null,1,1,1,32,1,1)")
    connection.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values(50,55,0,2,42,128,1,1,2,null)")
    connection.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values(30,40,0,2,null,1,1,99,1,1,1,32,1,1)")
    connection.commit()
    connection.close()

    output = extract(db)
    activity = [event for event in output["events"] if event["kind"] in ("kernel", "memcpy")]
    kernel = next(event for event in activity if event["kind"] == "kernel" and event["graph_node_id"] is None)
    memcpy = next(event for event in activity if event["kind"] == "memcpy")
    graph_kernel = next(event for event in activity if event["graph_node_id"] == 99)

    assert output["nvtx_event_count"] == 5
    assert any(event["phase_name"] == "prefill" for event in output["nvtx_events"])
    assert kernel["semantic_operator"].startswith("attention.qkv|")
    assert kernel["semantic_operator_id"] == "attention.qkv"
    assert kernel["semantic_layout"] == "contiguous"
    assert kernel["semantic_status"] == "matched"
    assert kernel["semantic_source"] == "NVTX_EVENTS/runtime_correlation_same_thread_containment"
    assert kernel["stage_evidence"] == "name_heuristic_only"
    assert memcpy["semantic_operator"].startswith("attention.qkv|")
    assert memcpy["semantic_layout"] == "contiguous"
    assert graph_kernel["semantic_status"] == "unknown"
    assert graph_kernel["semantic_reason"] == "graph_activity_without_runtime_nvtx_correlation"


def test_operator_id_excludes_shape_and_dtype_metadata():
    label = "operator:MUL_MAT:ffn_gate-7|shape=4864x8x1x1|type=f32"
    assert _operator_id(label) == "MUL_MAT:ffn_gate-7"


def test_request_lifecycle_marker_prefix_is_extracted():
    assert _explicit_request_marker("request_begin|slot=2|prompt_tokens=8") == "request_begin"
    assert _explicit_request_marker("first_token|slot=2") == "first_token"
    assert _explicit_request_marker("operator:attention.qkv") is None


def test_extract_preserves_request_marker_events(tmp_path: Path):
    db = tmp_path / "request.sqlite"
    connection = _db(db)
    connection.executemany("insert into NVTX_EVENTS values(?, ?, ?, ?)", [
        (10, 10, 7, "request_begin|slot=0|prompt_tokens=2|predict_tokens=8"),
        (30, 30, 7, "first_token|slot=0"),
        (70, 70, 7, "request_end|slot=0"),
    ])
    connection.commit()
    connection.close()
    output = extract(db)
    assert output["request_marker_count"] == 3
    assert [event["request_marker"] for event in output["request_markers"]] == [
        "request_begin", "first_token", "request_end"
    ]


def test_extracts_engine_token_boundary_markers_with_shape_fields(tmp_path: Path):
    db = tmp_path / "engine_boundaries.sqlite"
    connection = _db(db)
    connection.executemany("insert into NVTX_EVENTS values(?, ?, ?, ?)", [
        (10, 10, 7, "engine_request_begin|slot=0"),
        (20, 20, 7, "engine_token_begin|slot=0|token_index=1|token_count=1"),
        (30, 30, 7, "engine_token_end|slot=0|token_index=1|token_count=1"),
        (40, 40, 7, "engine_request_end|slot=0"),
    ])
    connection.commit()
    connection.close()
    output = extract(db)
    markers = output["request_markers"]
    assert [event["engine_boundary"] for event in markers] == [
        "engine_request_begin", "engine_token_begin", "engine_token_end", "engine_request_end"
    ]
    assert markers[1]["engine_token_index"] == "1"
    assert markers[1]["engine_token_count"] == "1"


def test_extracts_engine_compute_markers_separately_from_sampling_markers(tmp_path: Path):
    db = tmp_path / "engine_compute.sqlite"
    connection = _db(db)
    connection.executemany("insert into NVTX_EVENTS values(?, ?, ?, ?)", [
        (10, 10, 7, "engine_request_begin|slot=0"),
        (20, 20, 7, "engine_compute_begin|slot=0|token_index=1|token_count=1"),
        (90, 90, 7, "engine_compute_end|slot=0|token_index=1|token_count=1"),
        (95, 95, 7, "engine_token_begin|slot=0|token_index=1|token_count=1"),
        (96, 96, 7, "engine_token_end|slot=0|token_index=1|token_count=1"),
        (100, 100, 7, "engine_request_end|slot=0"),
    ])
    connection.commit()
    connection.close()
    output = extract(db)
    compute = [event for event in output["request_markers"]
               if event["engine_boundary"].startswith("engine_compute")]
    assert [event["engine_boundary"] for event in compute] == [
        "engine_compute_begin", "engine_compute_end"
    ]
    summary = output["engine_marker_summary"]
    assert summary["token_indices"] == ["1"]


def test_engine_marker_summary_separates_token_spans_from_inter_token_gaps(tmp_path: Path):
    db = tmp_path / "engine_gap.sqlite"
    connection = _db(db)
    connection.executemany("insert into NVTX_EVENTS values(?, ?, ?, ?)", [
        (10, 10, 7, "engine_request_begin|slot=0"),
        (20, 20, 7, "engine_token_begin|slot=0|token_index=1|token_count=1"),
        (30, 30, 7, "engine_token_end|slot=0|token_index=1|token_count=1"),
        (130, 130, 7, "engine_token_begin|slot=0|token_index=2|token_count=1"),
        (140, 140, 7, "engine_token_end|slot=0|token_index=2|token_count=1"),
        (150, 150, 7, "engine_request_end|slot=0"),
    ])
    connection.commit()
    connection.close()
    summary = extract(db)["engine_marker_summary"]
    assert summary["status"] == "complete"
    assert summary["token_indices"] == ["1", "2"]
    assert summary["token_intervals_ms"] == [pytest.approx(0.00001), pytest.approx(0.00001)]
    assert summary["inter_token_gaps_ms"] == [pytest.approx(0.0001)]
    assert summary["inter_token_begin_gaps_ms"] == [pytest.approx(0.00011)]
    assert summary["engine_request_interval_ms"] == pytest.approx(0.00014)
    assert summary["cost_inference"].startswith("forbidden")


def test_extracts_null_end_marks_separately_from_async_ranges(tmp_path: Path):
    db = tmp_path / "request_points.sqlite"
    connection = _db(db)
    connection.execute("alter table NVTX_EVENTS add column eventType integer")
    connection.executemany("insert into NVTX_EVENTS values(?, ?, ?, ?, ?)", [
        (10, None, 7, "request_begin|slot=0", 34),
        (11, 70, 7, "request_begin|slot=0", 60),
        (70, None, 7, "request_end|slot=0", 34),
    ])
    connection.commit()
    connection.close()
    output = extract(db, start_ns=0, end_ns=100)
    points = [event for event in output["request_markers"] if event["is_instant"]]
    assert [event["request_marker"] for event in points] == ["request_begin", "request_end"]
    assert all(event["duration_ns"] == 0 for event in points)
    assert output["request_markers"][1]["duration_ns"] == 59


def test_host_boundary_markers_keep_missing_boundaries_explicit(tmp_path: Path):
    db = tmp_path / "host.sqlite"
    connection = _db(db)
    connection.executemany("insert into NVTX_EVENTS values(?, ?, ?, ?)", [
        (10, None, 7, "http.handler.begin"),
        (20, None, 7, "slot.launch"),
        (30, None, 7, "response.queue"),
        (40, None, 7, "http.handler.end"),
    ])
    connection.commit()
    connection.close()
    output = extract_host_boundary_markers(db)
    assert output["counts"]["slot.launch"] == 1
    assert output["counts"]["json.tokenize"] == 0
    assert output["counts"]["sink.write"] == 0
    assert all(event["event_kind"] == "instant" for event in output["events"])
