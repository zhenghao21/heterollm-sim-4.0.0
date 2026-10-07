"""GET and save already-submitted browser jobs; never create simulation jobs."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / "docs/frontend_native_validation_2026-10-07"
DEFAULT_RAW_DIRECTORY = Path("F:/codex_project/_scratch/37-native-validation-raw")
TERMINAL = frozenset({"completed", "failed", "cancelled"})
KNOWN_STATES = TERMINAL | {"queued", "running", "cancelling"}
OMITTED_REPORT_FIELDS = ("visualization", "batch_history")
COMPACT_SCHEMA = "frontend-job-compact/v1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_index(directory):
    path = directory / "ui_jobs.json"
    if not path.exists():
        return []
    entries = read_json(path)
    if not isinstance(entries, list):
        raise ValueError("ui_jobs.json must contain an array")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("ui_jobs.json entries must be objects")
        prefix, job_id = entry.get("prefix"), entry.get("job_id")
        if not isinstance(prefix, str) or not re.fullmatch(r"ui_[A-Za-z0-9_-]+", prefix):
            raise ValueError("job prefix must be a plain ui_ filename stem")
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
            raise ValueError(f"invalid job_id for {prefix}")
        if prefix in seen:
            raise ValueError(f"duplicate prefix in ui_jobs.json: {prefix}")
        seen.add(prefix)
        url = urllib.parse.urlsplit(entry.get("base_url", "http://127.0.0.1:8765"))
        if url.scheme not in {"http", "https"} or not url.netloc or url.query or url.fragment:
            raise ValueError(f"invalid base_url for {prefix}")
    return entries


def fetch_job(entry, timeout):
    base = entry.get("base_url", "http://127.0.0.1:8765").rstrip("/")
    url = base + "/api/run-jobs/" + urllib.parse.quote(entry["job_id"], safe="")
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"unexpected HTTP {response.status}")
        return json.load(response)


def _save_json(path, value):
    # Readers should see either the previous complete snapshot or the new one.
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _progress(snapshot):
    progress = snapshot.get("progress") or {}
    ratio = progress.get("ratio")
    percent = f" {ratio * 100:.1f}%" if isinstance(ratio, (int, float)) else ""
    stage = progress.get("stage")
    return (" " + str(stage) if stage else "") + percent


def compact_terminal(snapshot, prefix, raw_directory=None):
    """Archive full terminal data before dropping only bulky replay details."""
    if snapshot.get("status") not in TERMINAL:
        return snapshot
    recording = snapshot.get("recording", {})
    if recording.get("schema") == COMPACT_SCHEMA:
        if not Path(recording["raw_archive_path"]).is_file():
            raise ValueError("compact result's original raw archive is missing")
        return snapshot
    report = snapshot.get("report")
    omitted = [key for key in OMITTED_REPORT_FIELDS if isinstance(report, dict) and key in report]
    if not omitted:
        return snapshot  # Small errors/cancellations already retain all data.
    job_id = snapshot.get("job_id")
    if (not re.fullmatch(r"ui_[A-Za-z0-9_-]+", prefix)
            or not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id)):
        raise ValueError("raw archive needs a safe prefix and job identity")
    raw_directory = Path(raw_directory or DEFAULT_RAW_DIRECTORY).resolve()
    raw_directory.mkdir(parents=True, exist_ok=True)
    archive = raw_directory / (prefix + "__" + job_id + ".json.gz")
    if archive.exists():
        with gzip.open(archive, "rt", encoding="utf-8") as handle:
            if json.load(handle) != snapshot:
                raise ValueError("existing raw archive differs; refusing to discard full result")
    else:
        descriptor, temporary_name = tempfile.mkstemp(prefix=archive.stem + ".", suffix=".tmp", dir=raw_directory)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
                json.dump(snapshot, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            with temporary.open("r+b") as handle:
                os.fsync(handle.fileno())
            with gzip.open(temporary, "rt", encoding="utf-8") as handle:
                if json.load(handle) != snapshot:
                    raise ValueError("raw archive round-trip differs from full result")
            # rename, rather than replace: never overwrite another collector's
            # completed archive on Windows.
            temporary.rename(archive)
        finally:
            temporary.unlink(missing_ok=True)
    compact_report = {key: value for key, value in report.items() if key not in omitted}
    return {**snapshot, "report": compact_report, "recording": {
        "schema": COMPACT_SCHEMA,
        "omitted_fields": ["report." + key for key in omitted],
        "raw_archive_path": str(archive),
        "raw_archive_format": "gzip-json",
        "raw_archive_verified": "full_json_round_trip_equal_before_compaction",
        "semantics": "All metrics and physical totals are unchanged; replay details are in the raw archive.",
    }}


def compact_result_file(path, raw_directory=None):
    existing = read_json(path)
    prefix = path.name.removesuffix("_result.json")
    compact = compact_terminal(existing, prefix, raw_directory)
    if compact is existing:
        return False
    # A renamed diagnostic or replacement must not be recreated after gzip.
    if not path.exists() or read_json(path).get("job_id") != existing.get("job_id"):
        raise ValueError("result changed during archival; compact result not saved")
    _save_json(path, compact)
    return True


def collect_once(directory, *, timeout=15.0, fetcher=None, raw_directory=None):
    """Preserve terminal results and return compact statuses for this index."""
    fetcher = fetcher or fetch_job
    statuses = []
    for entry in read_index(directory):
        prefix, job_id = entry["prefix"], entry["job_id"]
        path = directory / (prefix + "_result.json")
        status = {"prefix": prefix, "job_id": job_id, "terminal": False}
        try:
            existing = read_json(path) if path.exists() else None
            if existing is not None:
                if existing.get("job_id") != job_id:
                    raise ValueError("existing result belongs to another job; rename its attempt before collecting")
                if existing.get("status") in TERMINAL:
                    compact_result_file(path, raw_directory)
                    status.update(state=existing["status"], terminal=True, preserved=True,
                                  display=existing["status"] + " (saved)")
                    statuses.append(status)
                    continue
            snapshot = fetcher(entry, timeout)
            if not isinstance(snapshot, dict) or snapshot.get("job_id") != job_id:
                raise ValueError("server response job_id does not match requested job")
            state = snapshot.get("status")
            if state not in KNOWN_STATES:
                raise ValueError(f"server returned unknown job state {state!r}")
            if state == "completed" and not isinstance(snapshot.get("report"), dict):
                raise ValueError("completed server job has no report; preserving previous snapshot")
            # The user may rename cancelled diagnostics or add replacements
            # while a GET is in flight. Do not recreate a retired filename.
            current = next((item for item in read_index(directory) if item["prefix"] == prefix), None)
            if current is None or current["job_id"] != job_id:
                status.update(state="index_changed", display="index changed; response not saved")
            else:
                latest = read_json(path) if path.exists() else None
                if latest is not None and latest.get("job_id") != job_id:
                    raise ValueError("result changed to another job while GET was in flight")
                if latest is not None and latest.get("status") in TERMINAL:
                    status.update(state=latest["status"], terminal=True, preserved=True,
                                  display=latest["status"] + " (saved)")
                else:
                    snapshot = compact_terminal(snapshot, prefix, raw_directory)
                    current = next((item for item in read_index(directory) if item["prefix"] == prefix), None)
                    if current is None or current["job_id"] != job_id:
                        status.update(state="index_changed", display="index changed; archived response not saved")
                        statuses.append(status)
                        continue
                    _save_json(path, snapshot)
                    status.update(state=state, terminal=state in TERMINAL, saved=True,
                                  display=state + _progress(snapshot))
        except (OSError, ValueError, TypeError, urllib.error.URLError) as error:
            status.update(state="collection_error", error=f"{type(error).__name__}: {error}",
                          display=f"collection error; existing result preserved: {error}")
        statuses.append(status)
    return statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--raw-directory", type=Path, default=DEFAULT_RAW_DIRECTORY)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="collect the current index once (default)")
    mode.add_argument("--compact-existing", action="store_true",
                      help="archive and compact existing local terminal results only; no HTTP requests")
    mode.add_argument("--watch", nargs="?", const=5.0, type=float, metavar="SECONDS",
                      help="keep watching, including jobs added later; default interval 5 seconds")
    parser.add_argument("--until-terminal", action="store_true",
                        help="with --watch, exit when every job in a nonempty current index is terminal")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    if (args.watch is not None and args.watch <= 0) or args.timeout <= 0:
        parser.error("watch interval and timeout must be positive")
    if args.until_terminal and args.watch is None:
        parser.error("--until-terminal requires --watch")
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.compact_existing:
        for path in sorted(args.directory.glob("ui_*_result.json")):
            if compact_result_file(path, args.raw_directory):
                print(path.name + ": original archived; replay details compacted", flush=True)
        return 0
    displayed = {}
    index_error = None
    try:
        while True:
            try:
                statuses = collect_once(args.directory, timeout=args.timeout, raw_directory=args.raw_directory)
                index_error = None
            except (OSError, ValueError, TypeError) as error:
                message = f"index unreadable; no results changed: {error}"
                if message != index_error:
                    print(message, flush=True)
                index_error = message
                statuses = []
                if args.watch is None:
                    return 1
            for row in statuses:
                key = row["prefix"], row["job_id"]
                if displayed.get(key) != row["display"]:
                    print(row["prefix"] + ": " + row["display"], flush=True)
                    displayed[key] = row["display"]
            if args.watch is None:
                return 1 if any(row["state"] == "collection_error" for row in statuses) else 0
            if args.until_terminal and statuses and all(row["terminal"] for row in statuses):
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("Stopped collecting; saved results preserved.", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
