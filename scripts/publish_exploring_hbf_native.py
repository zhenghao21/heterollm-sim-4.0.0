"""Publish the native comparison to its existing website without running scans.

Credentials stay inside the Node helper, which reuses the current Netlify CLI
login. This program has no credential option and never changes permissions.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

from exploring_hbf_native_status import BEIJING, DEFAULT_DIRECTORY, ROOT, SCHEMA, write_snapshot

SCAN_ID = "exploring_hbf_native_20261027_pp_prefill_final4"
STATUS_PATH = ROOT / "tmp/native_comparison_publisher.json"
LOG_PATH = ROOT / "tmp/native_comparison_publisher.log"
LOCK_PATH = ROOT / "tmp/native_comparison_publisher.lock"
CLI_ROOT = Path("C:/Users/A/AppData/Local/npm-cache/_npx/da5c1b6ea715e8b4/node_modules/netlify-cli")
UPLOAD_KEY = "native-comparison/latest.json"


class PublisherConflict(RuntimeError):
    pass


class UploadError(RuntimeError):
    def __init__(self, name: str, *, code: str | None = None, status: int | None = None):
        self.safe_details = {"name": name, "code": code, "status": status}
        super().__init__(name)


class SinglePublisher:
    """One OS-released lock for this publication target; no PID-file guessing."""
    def __init__(self, path: Path = LOCK_PATH):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b" ")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise PublisherConflict("publisher already running") from None
        return self

    def __exit__(self, *_):
        self.handle.close()


def now() -> str:
    return datetime.now(BEIJING).isoformat()


def _safe_ack_error(value: Any) -> UploadError:
    value = value if isinstance(value, dict) else {}
    name = value.get("name")
    name = name if isinstance(name, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,47}", name) else "NodeUploadError"
    code = value.get("code")
    code = code if isinstance(code, str) and re.fullmatch(r"[A-Z_][A-Z0-9_]{0,31}", code) else None
    status = value.get("status")
    status = status if type(status) is int and 100 <= status <= 599 else None
    return UploadError(name, code=code, status=status)


def upload_snapshot(snapshot: dict[str, Any], *, node: str = "node", cli_root: Path = CLI_ROOT) -> dict[str, Any]:
    if snapshot.get("schema") != SCHEMA or snapshot.get("scan_id") != SCAN_ID:
        raise UploadError("InvalidSnapshot", code="INVALID_SNAPSHOT")
    command = [node, str(ROOT / "scripts/netlify_native_upload.mjs"), "--cli-root", str(cli_root)]
    options: dict[str, Any] = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    try:
        result = subprocess.run(command, input=json.dumps(snapshot, ensure_ascii=False, allow_nan=False),
            text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=ROOT, timeout=30, check=False, **options)
    except subprocess.TimeoutExpired:
        # subprocess.run kills and waits for the timed-out Node process.
        raise UploadError("UploadTimeout", code="UPLOAD_TIMEOUT") from None
    except OSError:
        raise UploadError("NodeUnavailable", code="NODE_UNAVAILABLE") from None
    try:
        acknowledgement = json.loads(result.stdout)
    except (ValueError, TypeError):
        raise UploadError("InvalidAcknowledgement", code="INVALID_ACK") from None
    if (result.returncode != 0 or not isinstance(acknowledgement, dict)
        or acknowledgement.get("ok") is not True or acknowledgement.get("verified") is not True):
        raise _safe_ack_error(acknowledgement)
    if acknowledgement.get("key") != UPLOAD_KEY or acknowledgement.get("observed_at") != snapshot.get("observed_at"):
        raise UploadError("ReadbackMismatch", code="READBACK_MISMATCH")
    status = acknowledgement.get("http_status")
    return {"verified": True, "observed_at": snapshot["observed_at"],
            "http_status": status if type(status) is int and 100 <= status <= 599 else None}


def _write_status(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".native-publisher-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def publish_once(directory: Path, *, previous: dict[str, Any] | None = None,
                 node: str = "node", cli_root: Path = CLI_ROOT) -> dict[str, Any]:
    previous = previous or {}
    state = {"pid": os.getpid(), "state": "publishing", "scan_id": SCAN_ID, "updated_at": now(),
        "last_success_at": previous.get("last_success_at"),
        "last_success_observed_at": previous.get("last_success_observed_at"),
        "consecutive_failures": previous.get("consecutive_failures", 0), "http_status": None,
        "verified": False, "error": None, "key": UPLOAD_KEY}
    try:
        snapshot = write_snapshot(directory)
        acknowledgement = upload_snapshot(snapshot, node=node, cli_root=cli_root)
        state.update(state="healthy", updated_at=now(), last_success_at=now(),
            last_success_observed_at=snapshot["observed_at"], consecutive_failures=0,
            http_status=acknowledgement["http_status"], verified=True,
            completed_cases=len(snapshot["completed_results"]))
    except UploadError as exc:
        state.update(state="error", error=exc.safe_details,
            http_status=exc.safe_details.get("status"), consecutive_failures=state["consecutive_failures"] + 1)
    except Exception:
        # Snapshot/configuration failures must not dump file data or arbitrary exception text.
        state.update(state="error", error={"name": "SnapshotError", "code": "SNAPSHOT_ERROR", "status": None},
            consecutive_failures=state["consecutive_failures"] + 1)
    state["updated_at"] = now()
    return state


def run_publisher(directory: Path, *, interval: float = 15., once: bool = False,
                  node: str = "node", cli_root: Path = CLI_ROOT,
                  status_path: Path = STATUS_PATH, lock_path: Path = LOCK_PATH) -> int:
    with SinglePublisher(lock_path):
        state: dict[str, Any] = {}
        while True:
            started = time.monotonic()
            state = publish_once(directory, previous=state, node=node, cli_root=cli_root)
            _write_status(status_path, state)
            print(json.dumps(state, ensure_ascii=False), flush=True)
            if once:
                return 0 if state["state"] == "healthy" else 2
            time.sleep(max(0., interval - (time.monotonic() - started)))


def start_background(args: argparse.Namespace) -> dict[str, Any]:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--directory", str(args.directory.resolve()),
        "--interval", str(args.interval), "--node", args.node, "--cli-root", str(args.cli_root)]
    if args.once:
        command.append("--once")
    options: dict[str, Any] = {"cwd": str(ROOT), "stdin": subprocess.DEVNULL,
        "stderr": subprocess.STDOUT, "close_fds": True}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    else:
        options["start_new_session"] = True
    with LOG_PATH.open("ab") as log:
        process = subprocess.Popen(command, stdout=log, **options)
    return {"state": "background_started", "launcher_pid": process.pid,
        "status_file": str(STATUS_PATH), "log": str(LOG_PATH)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--interval", type=float, default=15.)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--node", default="node")
    parser.add_argument("--cli-root", type=Path, default=CLI_ROOT)
    args = parser.parse_args(argv)
    if args.interval < 1:
        parser.error("interval must be at least one second")
    if args.background:
        print(json.dumps(start_background(args), ensure_ascii=False), flush=True)
        return 0
    try:
        return run_publisher(args.directory, interval=args.interval, once=args.once,
            node=args.node, cli_root=args.cli_root)
    except PublisherConflict:
        print(json.dumps({"state": "already_running", "name": "PublisherConflict"}), flush=True)
        return 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
