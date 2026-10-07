from importlib.util import module_from_spec, spec_from_file_location
import gzip
import json
from pathlib import Path
import urllib.error

import pytest


_SPEC = spec_from_file_location("frontend_job_collection", Path(__file__).parents[1] / "tools/collect_frontend_jobs.py")
assert _SPEC is not None and _SPEC.loader is not None
collector = module_from_spec(_SPEC)
_SPEC.loader.exec_module(collector)


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def index(directory):
    write(directory / "ui_jobs.json", [{"prefix": "ui_test", "job_id": "abc"}])


def test_saved_terminal_survives_old_server_disappearance(tmp_path):
    index(tmp_path)
    terminal = {"job_id": "abc", "status": "completed", "report": {"summary": {}}}
    write(tmp_path / "ui_test_result.json", terminal)

    def gone(*args):
        raise AssertionError("saved terminal jobs should not need a GET")

    status = collector.collect_once(tmp_path, fetcher=gone)[0]
    assert status["terminal"] and status["preserved"]
    assert collector.read_json(tmp_path / "ui_test_result.json") == terminal


def test_404_preserves_last_running_snapshot(tmp_path):
    index(tmp_path)
    previous = {"job_id": "abc", "status": "running"}
    write(tmp_path / "ui_test_result.json", previous)

    def gone(*args):
        raise urllib.error.HTTPError("http://127.0.0.1/abc", 404, "not found", {}, None)

    assert collector.collect_once(tmp_path, fetcher=gone)[0]["state"] == "collection_error"
    assert collector.read_json(tmp_path / "ui_test_result.json") == previous


def test_collects_matching_completed_job(tmp_path):
    index(tmp_path)
    result = {"job_id": "abc", "status": "completed", "report": {"summary": {}}}
    status = collector.collect_once(tmp_path, fetcher=lambda *_: result)[0]
    assert status["terminal"] and status["saved"]
    assert collector.read_json(tmp_path / "ui_test_result.json") == result


def test_reused_prefix_does_not_overwrite_previous_job(tmp_path):
    index(tmp_path)
    previous = {"job_id": "old", "status": "completed", "report": {}}
    write(tmp_path / "ui_test_result.json", previous)
    status = collector.collect_once(tmp_path, fetcher=lambda *_: {})[0]
    assert status["state"] == "collection_error"
    assert collector.read_json(tmp_path / "ui_test_result.json") == previous


def test_retired_index_entry_is_not_recreated_after_get(tmp_path):
    index(tmp_path)

    def fetch(*args):
        write(tmp_path / "ui_jobs.json", [])
        return {"job_id": "abc", "status": "running"}

    assert collector.collect_once(tmp_path, fetcher=fetch)[0]["state"] == "index_changed"
    assert not (tmp_path / "ui_test_result.json").exists()


def test_concurrent_terminal_save_is_not_replaced_by_older_get(tmp_path):
    index(tmp_path)
    terminal = {"job_id": "abc", "status": "completed", "report": {}}

    def fetch(*args):
        write(tmp_path / "ui_test_result.json", terminal)
        return {"job_id": "abc", "status": "running"}

    assert collector.collect_once(tmp_path, fetcher=fetch)[0]["preserved"]
    assert collector.read_json(tmp_path / "ui_test_result.json") == terminal


def full_terminal():
    return {"job_id": "abc", "status": "completed", "report": {
        "summary": {"finished": 1}, "requests": [{"engine_tpot_ns": 10}],
        "dram_traffic": {"physical_read_bytes": 1000},
        "batch_trace_index": {"batches": [{"event_count": 128}]},
        "visualization": {"events": [{"details": "x" * 2000}]},
        "batch_history": [{"cost": {"details": "x" * 2000}}],
    }}


def test_full_terminal_archived_before_compaction_and_metrics_unchanged(tmp_path):
    index(tmp_path)
    original = full_terminal()
    status = collector.collect_once(tmp_path, fetcher=lambda *_: original,
                                    raw_directory=tmp_path / "raw")[0]
    assert status["terminal"] and status["saved"]
    compact = collector.read_json(tmp_path / "ui_test_result.json")
    assert compact["report"] == {k: v for k, v in original["report"].items()
                                 if k not in {"visualization", "batch_history"}}
    assert compact["recording"]["omitted_fields"] == ["report.visualization", "report.batch_history"]
    archive = Path(compact["recording"]["raw_archive_path"])
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        assert json.load(handle) == original
    modified = archive.stat().st_mtime_ns
    collector.collect_once(tmp_path, fetcher=lambda *_: pytest.fail("terminal must not be fetched"),
                           raw_directory=tmp_path / "raw")
    assert archive.stat().st_mtime_ns == modified


def test_archive_failure_preserves_original_full_file(tmp_path):
    path = tmp_path / "ui_test_result.json"
    original = full_terminal()
    write(path, original)
    raw = tmp_path / "raw"
    raw.mkdir()
    with gzip.open(raw / "ui_test__abc.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"wrong": "existing archive"}, handle)
    with pytest.raises(ValueError, match="differs"):
        collector.compact_result_file(path, raw)
    assert collector.read_json(path) == original


def test_running_snapshot_never_archived(tmp_path):
    original = {**full_terminal(), "status": "running"}
    assert collector.compact_terminal(original, "ui_test", tmp_path / "raw") is original
    assert not (tmp_path / "raw").exists()
