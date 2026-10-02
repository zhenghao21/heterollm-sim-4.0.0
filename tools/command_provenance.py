"""Run one command and persist return-code/output provenance."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def run_command(
    command: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Capture a child command's outputs and return a JSON-safe record."""

    started = time.perf_counter()
    timed_out = False
    stdout = b""
    stderr = b""
    returncode: int | None = None
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
        stdout = completed.stdout or b""
        stderr = completed.stderr or b""
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
    return {
        "schema": "heterollm.command_provenance/v1",
        "command": [str(item) for item in command],
        "cwd": str(Path(cwd or Path.cwd()).resolve()),
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_seconds": time.perf_counter() - started,
        "stdout_sha256": _sha256(stdout),
        "stderr_sha256": _sha256(stderr),
        "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr),
        "stdout_base64": base64.b64encode(stdout).decode("ascii"),
        "stderr_base64": base64.b64encode(stderr).decode("ascii"),
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
    }


def write_record(record: Mapping[str, Any], output_path: str | os.PathLike[str]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(record), ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        "{}  {}\n".format(digest, path.name), encoding="ascii"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="JSON provenance output path")
    parser.add_argument("--cwd", default=None, help="child working directory")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a command is required after --")
    record = run_command(command, cwd=args.cwd, timeout_seconds=args.timeout)
    write_record(record, args.output)
    return int(record["returncode"] if record["returncode"] is not None else 124)


if __name__ == "__main__":
    raise SystemExit(main())
