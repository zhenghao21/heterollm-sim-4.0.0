import sqlite3

from tools.extract_cuda_api_phase import extract


def test_extract_cuda_api_phase_requires_same_thread_containment(tmp_path):
    db = tmp_path / "trace.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE StringIds(id INTEGER, value TEXT);"
        "CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, globalTid INTEGER);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, eventClass INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER, returnValue INTEGER, callchainId INTEGER);"
    )
    conn.executemany("INSERT INTO StringIds VALUES (?, ?)", [
        (1, "cudaLaunchKernel_v7000"), (2, "cudaStreamSynchronize_v3020")
    ])
    conn.execute("INSERT INTO NVTX_EVENTS VALUES (100, 200, 'phase:prefill', 7)")
    conn.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 0, 7, 0, ?, 0, 0)", [
        (110, 120, 1),  # contained launch
        (190, 210, 2),  # crosses the phase boundary
    ])
    conn.commit(); conn.close()
    result = extract(db)
    assert result["api"]["prefill"]["launch"] == {"count": 1, "total_ns": 10}
    assert result["excluded_boundary_calls"] == 1


def test_extract_resolves_phase_text_id_and_ignores_instant_phase_mark(tmp_path):
    db = tmp_path / "textid.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE StringIds(id INTEGER, value TEXT);"
        "CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, eventType INTEGER, text TEXT, textId INTEGER, globalTid INTEGER);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, eventClass INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER, returnValue INTEGER, callchainId INTEGER);"
    )
    conn.executemany("INSERT INTO StringIds VALUES (?, ?)", [
        (1, "phase:decode"), (2, "cudaLaunchKernel_v7000")
    ])
    # The range label is carried only by textId, as in some Nsight exports.
    conn.execute("INSERT INTO NVTX_EVENTS VALUES (100, 200, 60, NULL, 1, 7)")
    # An instant marker with the same text must not be treated as a range.
    conn.execute("INSERT INTO NVTX_EVENTS VALUES (100, NULL, 34, NULL, 1, 7)")
    conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (110, 120, 0, 7, 41, 2, 0, 0)")
    conn.commit(); conn.close()
    result = extract(db)
    assert result["phase_ranges"] == {"prefill": 0, "decode": 1}
    assert result["api"]["decode"]["launch"] == {"count": 1, "total_ns": 10}
    assert result["instant_phase_markers"] == 1
    assert result["excluded_boundary_calls"] == 0
