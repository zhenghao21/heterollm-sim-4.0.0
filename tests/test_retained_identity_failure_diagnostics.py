"""Small-file identity failure diagnostics; no native or simulator execution."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import predict_stable_native_dataset as api


CHUNK_BYTES = 4 * 1024 * 1024
STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
CHECKS = ("length_matches", "handle_identity_unchanged", "path_identity_matches_opened_file", "sha256_matches")


@pytest.fixture
def exercise(tmp_path, monkeypatch):
    """Perturb one original gate observation without a second file read."""
    def run(*, short_read=False, expect_short_digest=False, wrong_digest=False,
            changed_handle=False, changed_path=False, path_stat_error=None,
            vary_ctime=False):
        payload = b"small synthetic weight body for identity diagnostics"
        path = tmp_path / "weights.gguf"
        path.write_bytes(payload)
        original_hash = hashlib.sha256
        original_open, original_stat, original_fstat = Path.open, Path.stat, os.fstat
        delivered = payload[:-1] if short_read else payload
        expected_sha = original_hash(delivered if expect_short_digest else payload).hexdigest()
        if wrong_digest:
            expected_sha = "0" * 64
        ref = {"path": str(path.resolve()), "bytes": len(payload), "sha256": expected_sha}
        actual_sha = original_hash(delivered).hexdigest()
        counts = {"opens": 0, "fstats": 0, "post_read_path_stats": 0,
                  "hashes": 0, "hexdigests": 0, "read_sizes": [], "updates": []}
        state = {"fd": None, "closed": False}
        observations = {}

        def stat_copy(value, **changes):
            fields = {name: getattr(value, name) for name in (*STAT_FIELDS, "st_ctime_ns")}
            fields.update(changes)
            return SimpleNamespace(**fields)

        class Stream:
            def __init__(self, inner):
                self.inner = inner
                state["fd"] = inner.fileno()
            def __enter__(self):
                return self
            def __exit__(self, *args):
                result = self.inner.__exit__(*args)
                state["closed"] = True
                return result
            def fileno(self):
                return self.inner.fileno()
            def read(self, size):
                counts["read_sizes"].append(size)
                data = self.inner.read(size)
                return data[:-1] if data and short_read else data

        def open_file(self, *args, **kwargs):
            inner = original_open(self, *args, **kwargs)
            if self == path and args == ("rb",):
                counts["opens"] += 1
                return Stream(inner)
            return inner

        def handle_stat(fd):
            value = original_fstat(fd)
            if fd == state["fd"]:
                counts["fstats"] += 1
                first = counts["fstats"] == 1
                changes = {}
                if changed_handle and first:
                    changes["st_ino"] = value.st_ino + 1
                if vary_ctime:
                    changes["st_ctime_ns"] = counts["fstats"]
                value = stat_copy(value, **changes)
                observations["fstat_before" if first else "fstat_after"] = value
            return value

        def path_stat(self, *args, **kwargs):
            if self == path and state["closed"]:
                counts["post_read_path_stats"] += 1
                if path_stat_error is not None:
                    raise path_stat_error
                value = original_stat(self, *args, **kwargs)
                changes = {"st_mtime_ns": value.st_mtime_ns + 1} if changed_path else {}
                if vary_ctime:
                    changes["st_ctime_ns"] = 3
                value = stat_copy(value, **changes)
                observations["path_stat_after"] = value
                return value
            return original_stat(self, *args, **kwargs)

        class Digest:
            def __init__(self, *args, **kwargs):
                counts["hashes"] += 1
                self.inner = original_hash(*args, **kwargs)
            def update(self, chunk):
                counts["updates"].append(len(chunk))
                self.inner.update(chunk)
            def hexdigest(self):
                counts["hexdigests"] += 1
                return self.inner.hexdigest()

        result, error, diagnostic = None, None, None
        with monkeypatch.context() as patches:
            patches.setattr(Path, "open", open_file)
            patches.setattr(Path, "stat", path_stat)
            patches.setattr(os, "fstat", handle_stat)
            patches.setattr(hashlib, "sha256", Digest)
            try:
                result = api.verify_retained_model_identities([ref])
            except ValueError as exc:
                error = exc
                prefix = "retained KV full model SHA256/identity mismatch: " + ref["path"] + "; diagnostic="
                assert str(exc).startswith(prefix)
                diagnostic = json.loads(str(exc)[len(prefix):])
        return SimpleNamespace(ref=ref, result=result, error=error, diagnostic=diagnostic,
            counts=counts, actual_sha=actual_sha, actual_bytes=len(delivered),
            observations=observations)
    return run


def assert_one_pass(result):
    assert result.counts == {"opens": 1, "fstats": 2, "post_read_path_stats": 1,
        "hashes": 1, "hexdigests": 1, "read_sizes": [CHUNK_BYTES, CHUNK_BYTES],
        "updates": [result.actual_bytes]}


def assert_diagnostic(result, failed):
    record = result.diagnostic
    assert record["schema"] == "retained-model-identity-check/v1"
    assert record["path"] == result.ref["path"]
    assert record["read_chunk_bytes"] == CHUNK_BYTES
    assert record["expected"] == {key: result.ref[key] for key in ("bytes", "sha256")}
    assert record["actual"]["bytes"] == result.actual_bytes
    assert record["actual"]["sha256"] == result.actual_sha
    assert record["checks"] == {name: name not in failed for name in CHECKS}
    assert record["failed_checks"] == [name for name in CHECKS if name in failed]
    started = datetime.fromisoformat(record["started_utc"])
    finished = datetime.fromisoformat(record["finished_utc"])
    assert started.tzinfo is not None and finished >= started
    for label, observed in result.observations.items():
        actual = record["actual"][label]
        assert set(actual) == {*STAT_FIELDS, "observed_utc"}
        assert {name: actual[name] for name in STAT_FIELDS} == {
            name: getattr(observed, name) for name in STAT_FIELDS}
        assert started <= datetime.fromisoformat(actual["observed_utc"]) <= finished
    assert_one_pass(result)


def test_success_returns_original_canonical_ref_with_one_read_and_digest(exercise):
    result = exercise()
    assert result.error is None
    assert result.result == [result.ref]
    assert_one_pass(result)


@pytest.mark.parametrize("options,failed", [
    ({"short_read": True, "expect_short_digest": True}, "length_matches"),
    ({"changed_handle": True}, "handle_identity_unchanged"),
    ({"changed_path": True}, "path_identity_matches_opened_file"),
    ({"wrong_digest": True}, "sha256_matches"),
])
def test_each_failure_reports_actual_values_without_short_circuiting(exercise, options, failed):
    result = exercise(**options)
    assert isinstance(result.error, ValueError)
    assert_diagnostic(result, {failed})
    assert result.diagnostic["path_stat_error"] is None


def test_joint_failure_retains_all_failed_predicates(exercise):
    result = exercise(short_read=True, changed_handle=True, changed_path=True)
    assert_diagnostic(result, set(CHECKS))
    assert result.diagnostic["path_stat_error"] is None


@pytest.mark.parametrize("stat_error", [FileNotFoundError("path disappeared"), PermissionError("path denied")])
def test_final_path_stat_error_preserves_completed_read(exercise, stat_error):
    result = exercise(path_stat_error=stat_error)
    assert_diagnostic(result, {"path_identity_matches_opened_file"})
    assert result.error.__cause__ is stat_error
    assert result.diagnostic["actual"]["path_stat_after"] is None
    failure = result.diagnostic["path_stat_error"]
    assert failure["type"] == type(stat_error).__name__
    assert failure["message"] == str(stat_error)
    stamp = datetime.fromisoformat(failure["observed_utc"])
    assert datetime.fromisoformat(result.diagnostic["started_utc"]) <= stamp <= datetime.fromisoformat(result.diagnostic["finished_utc"])


def test_ctime_difference_does_not_change_original_acceptance(exercise):
    result = exercise(vary_ctime=True)
    assert result.error is None
    assert result.result == [result.ref]
    assert_one_pass(result)
