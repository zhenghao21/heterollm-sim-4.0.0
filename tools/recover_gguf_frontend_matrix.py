"""Recover already-submitted frontend jobs through serial, GET-only observations.

Responses stream to disk before one snapshot is parsed at a time. Completed or
failed results retain the original JSON bytes, without another serialization of
the potentially large report. This helper never imports or submits scenarios.
"""

from __future__ import annotations

import argparse
import errno
import gc
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib.parse import quote, urlsplit
from urllib.request import ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "gguf_preset_native_validation_2026-10-08"
RECOVERY_METHOD = "serial read-only API after observer restart"
TERMINAL = {"completed", "failed", "cancelled"}
STATUSES = {"queued", "running"} | TERMINAL


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit(**fields: Any) -> None:
    print(json.dumps(fields, ensure_ascii=False), flush=True)


def retry_file(operation: Callable[[], Any]) -> Any:
    for attempt in range(6):
        try:
            return operation()
        except OSError as error:
            transient = (
                error.errno in {errno.EBUSY, errno.EPERM, errno.EACCES}
                or getattr(error, "winerror", None) in {5, 32, 33}
                or "UNKNOWN" in str(error)
            )
            if not transient or attempt == 5:
                raise
            time.sleep(min(0.2, 0.025 * 2**attempt))


def temporary_path(target: Path) -> Path:
    return target.with_name(f"{target.name}.{os.getpid()}.{time.time_ns()}.tmp")


def cleanup(path: Path) -> None:
    try:
        retry_file(lambda: path.unlink(missing_ok=True))
    except OSError as error:
        emit(status="temporary_file_cleanup_failed", path=str(path), error=str(error))


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = temporary_path(path)
    try:
        # Only the small manifest is serialized. Full result bytes are renamed.
        data = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        retry_file(lambda: temporary.write_text(data, encoding="utf-8"))
        retry_file(lambda: os.replace(temporary, path))
    finally:
        cleanup(temporary)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def output_directory(value: str) -> Path:
    if not value.strip():
        raise argparse.ArgumentTypeError("output directory must not be empty")
    return Path(value)


def validate_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = manifest.get("runs")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest has no frontend jobs to recover")
    seen: set[tuple[str, str]] = set()
    seen_jobs: set[str] = set()
    for record in records:
        case_id, mode = record.get("case_id", ""), record.get("mode", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", case_id) or mode not in {"off", "on"}:
            raise ValueError("Invalid case/mode in manifest")
        key = case_id, mode
        if key in seen:
            raise ValueError(f"Multiple current attempts for {case_id}/{mode}; inspect manually")
        seen.add(key)
        job_id = record.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(f"{case_id}/{mode} has no recorded job_id; never resubmitting")
        if job_id in seen_jobs:
            raise ValueError(f"Repeated job_id for {case_id}/{mode}; inspect manually")
        seen_jobs.add(job_id)
        url = urlsplit(record.get("url", ""))
        if (
            url.scheme != "http"
            or url.hostname != "127.0.0.1"
            or url.port not in range(8794, 8814)
            or url.path != "/"
            or url.query
            or url.fragment
            or url.username
            or url.password
        ):
            raise ValueError(f"Invalid dedicated recovery URL for {case_id}/{mode}")
        created_path = Path(record["job_created_path"])
        if read_json(created_path).get("job_id") != job_id:
            raise ValueError(f"Original creation response does not match {case_id}/{mode}")
        # Original frontend semantic checks are retained; do not reparse scenarios
        # or submit another job. Missing originals require manual investigation.
        if not Path(record["submission_path"]).is_file():
            raise ValueError(f"Original submission is missing for {case_id}/{mode}")
        if record.get("status") == "completed" and not Path(record.get("result_path", "")).is_file():
            raise ValueError(f"Completed result is missing for {case_id}/{mode}; inspect manually")
    return records


def fetch_snapshot(opener: Any, record: dict[str, Any], output: Path, timeout: float) -> tuple[Path, dict[str, Any]]:
    result_path = output / f"ui_{record['case_id']}_graph_{record['mode']}_result.json"
    temporary = temporary_path(result_path)
    payload = None
    try:
        url = record["url"] + "api/run-jobs/" + quote(record["job_id"], safe="")
        request = Request(url, method="GET", headers={"Accept": "application/json"})
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise ValueError(f"Read-only recovery returned HTTP {response.status}")
            with retry_file(lambda: temporary.open("wb")) as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        payload = read_json(temporary)
        if payload.get("job_id") != record["job_id"]:
            raise ValueError("Recovery response belongs to a different job")
        if payload.get("status") not in STATUSES:
            raise ValueError("Recovery response has an unknown job status")
        summary = {key: payload.get(key) for key in ("job_id", "status", "progress", "error", "finished_at")}
        summary["report_ready"] = payload.get("report") is not None
        if summary["status"] == "completed" and not summary["report_ready"]:
            raise ValueError("Completed backend job has no report")
        # Keep only small top-level fields. No second full report/string is built.
        return temporary, summary
    except Exception:
        cleanup(temporary)
        raise
    finally:
        del payload
        gc.collect()


def observe(opener: Any, record: dict[str, Any], output: Path, timeout: float) -> bool:
    temporary, summary = fetch_snapshot(opener, record, output, timeout)
    try:
        terminal = summary["status"] in TERMINAL
        if terminal:
            result_path = output / f"ui_{record['case_id']}_graph_{record['mode']}_result.json"
            retry_file(lambda: os.replace(temporary, result_path))
            record["result_path"] = str(result_path)
            record["finished_at"] = summary["finished_at"] or now()
            record["recovered_report_rendering_checked"] = False
        record["status"] = summary["status"]
        record["last_progress"] = summary["progress"]
        record["observer_recovery"] = RECOVERY_METHOD
        record["observer_status"] = "finished" if terminal else "polling"
        record["last_observed_at"] = now()
        record["recovery_read_failures"] = 0
        record.pop("observer_error", None)
        record.pop("simulation_status", None)
        if summary["error"] is not None:
            record["error"] = summary["error"]
        else:
            record.pop("error", None)
        progress_path = Path(record.get("progress_path") or output / f"ui_{record['case_id']}_graph_{record['mode']}_progress.jsonl")
        event = {"observed_at": now(), "observer_recovery": RECOVERY_METHOD, **summary}
        # One line contains progress/status only; full reports stay on disk.
        try:
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as error:
            emit(case_id=record["case_id"], mode=record["mode"], status="progress_log_write_failed", error=str(error))
        emit(case_id=record["case_id"], mode=record["mode"], job_id=record["job_id"], status=record["status"], progress=summary["progress"])
        return terminal
    finally:
        cleanup(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--output", dest="output", type=output_directory, default=DEFAULT_OUTPUT,
                        help="Existing validation directory containing ui_runs.json (--output remains an alias)")
    parser.add_argument("--once", action="store_true", help="Observe each unfinished recorded job once, then exit")
    parser.add_argument("--interval", type=float, default=30.0, help="Seconds between serial polling rounds")
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout in seconds")
    args = parser.parse_args(argv)
    if any(not math.isfinite(value) or value <= 0 for value in (args.interval, args.timeout)):
        parser.error("interval and timeout must be finite and positive")
    args.output = args.output.resolve()
    if not args.output.is_dir():
        parser.error("output directory must be an existing prepared validation directory")
    manifest_path = args.output / "ui_runs.json"
    manifest = read_json(manifest_path)
    records = validate_records(manifest)
    finished = {r["job_id"] for r in records if r["status"] == "completed"}
    opener = build_opener(ProxyHandler({}))
    manifest["recovery_methodology"] = (
        "All jobs were submitted through the frontend. This observer only performs serial GET requests on original job IDs. "
        "Each HTTP response streams to disk; one snapshot at a time is parsed, its report released, and terminal raw bytes retained without reserialization. "
        "Original submission/job_created files are preserved. Recovered report rendering is not revalidated."
    )
    emit(status="read_only_recovery_started", recorded_jobs=len(records), already_completed=len(finished), once=args.once)
    while len(finished) < len(records):
        for record in records:
            if record["job_id"] in finished:
                continue
            try:
                if observe(opener, record, args.output, args.timeout):
                    finished.add(record["job_id"])
            except Exception as error:
                record["observer_status"] = "failed"
                record["observer_error"] = f"{type(error).__name__}: {error}"
                record["recovery_read_failures"] = record.get("recovery_read_failures", 0) + 1
                emit(case_id=record["case_id"], mode=record["mode"], job_id=record["job_id"], status="observer_read_failed", simulation_status=record["status"], error=record["observer_error"], automatic_resubmission=False)
            manifest["updated_at"] = now()
            try:
                save_manifest(manifest_path, manifest)
            except OSError as error:
                # Other backends keep running and can still be observed. Do not
                # confuse a local write failure with simulation failure.
                emit(status="manifest_write_failed", error=str(error), automatic_resubmission=False)
            gc.collect()
        if args.once or len(finished) == len(records):
            break
        time.sleep(args.interval)
    # Retry persistence once at shutdown; if it still fails, exit unsuccessfully.
    save_manifest(manifest_path, manifest)
    failed = [r for r in records if r["status"] in {"failed", "cancelled"} or r.get("observer_error")]
    emit(status="recovery_round_finished" if args.once else "recovery_finished", completed=sum(r["status"] == "completed" for r in records), terminal=len(finished), recorded_jobs=len(records), failed_or_unreadable=len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
