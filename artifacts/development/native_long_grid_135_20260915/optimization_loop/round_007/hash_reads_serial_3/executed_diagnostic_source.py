"""Read-only SHA-256 evidence collection; never mutates the source file.

Each pass uses a fresh Python ``open(path, "rb")`` and fresh digest. The chunk
log records every successful read, including the final partial chunk. The
run is serial. Durations are diagnostic wall times, not benchmark data.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CHUNK_BYTES = 1024 * 1024


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def dump_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def stat_identity(st):
    return {
        "st_dev": st.st_dev, "st_ino": st.st_ino, "st_mode": st.st_mode,
        "st_size": st.st_size, "st_mtime_ns": st.st_mtime_ns,
        "st_ctime_ns": st.st_ctime_ns,
        "st_file_attributes": getattr(st, "st_file_attributes", None),
    }


def native_handle_identity(stream):
    if os.name != "nt":
        return {"available": False, "reason": "not_windows"}
    import msvcrt
    from ctypes import wintypes

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_info = kernel32.GetFileInformationByHandle
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ByHandleFileInformation)]
    get_info.restype = wintypes.BOOL
    info = ByHandleFileInformation()
    handle = msvcrt.get_osfhandle(stream.fileno())
    if not get_info(wintypes.HANDLE(handle), ctypes.byref(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "available": True,
        "volume_serial_number": info.dwVolumeSerialNumber,
        "file_index": (info.nFileIndexHigh << 32) | info.nFileIndexLow,
        "file_size": (info.nFileSizeHigh << 32) | info.nFileSizeLow,
        "creation_filetime": (info.ftCreationTime.dwHighDateTime << 32) | info.ftCreationTime.dwLowDateTime,
        "last_write_filetime": (info.ftLastWriteTime.dwHighDateTime << 32) | info.ftLastWriteTime.dwLowDateTime,
        "attributes": info.dwFileAttributes,
        "number_of_links": info.nNumberOfLinks,
    }


def collect_pass(target, expected_sha, output_dir, ordinal):
    start = time.perf_counter_ns()
    chunk_path = output_dir / f"pass_{ordinal:02d}.chunks.jsonl"
    record = {
        "pass": ordinal, "pid": os.getpid(), "started_at_utc": utc_now(),
        "target": str(target), "expected_sha256": expected_sha,
        "chunk_bytes": CHUNK_BYTES, "chunk_log": chunk_path.name,
        "bytes_read": 0, "chunk_count": 0, "eof_observed": False,
        "status": "failed", "errors": [],
    }
    digest = hashlib.sha256()
    try:
        record["path_before"] = stat_identity(target.stat())
        with target.open("rb") as source, chunk_path.open("x", encoding="utf-8", newline="\n") as chunk_stream:
            record["handle_before"] = stat_identity(os.fstat(source.fileno()))
            record["native_handle_before"] = native_handle_identity(source)
            while True:
                block = source.read(CHUNK_BYTES)
                if not block:
                    record["eof_observed"] = True
                    break
                chunk_record = {
                    "chunk_index": record["chunk_count"],
                    "offset_bytes": record["bytes_read"],
                    "length_bytes": len(block),
                    "sha256": hashlib.sha256(block).hexdigest(),
                }
                digest.update(block)
                chunk_stream.write(json.dumps(chunk_record, separators=(",", ":")) + "\n")
                record["bytes_read"] += len(block)
                record["chunk_count"] += 1
            record["handle_after"] = stat_identity(os.fstat(source.fileno()))
            record["native_handle_after"] = native_handle_identity(source)
        record["path_after"] = stat_identity(target.stat())
        record["actual_sha256"] = digest.hexdigest()
        record["sha256_matches_expected"] = record["actual_sha256"] == expected_sha
        record["path_stat_unchanged"] = record["path_before"] == record["path_after"]
        record["handle_stat_unchanged"] = record["handle_before"] == record["handle_after"]
        record["path_and_handle_identity_match"] = (
            record["path_before"] == record["handle_before"]
            and record["path_after"] == record["handle_after"]
        )
        record["native_handle_unchanged"] = record["native_handle_before"] == record["native_handle_after"]
        record["bytes_match_all_recorded_sizes"] = all(
            record["bytes_read"] == record[key]["st_size"]
            for key in ("path_before", "path_after", "handle_before", "handle_after")
        )
        checks = (
            "sha256_matches_expected", "path_stat_unchanged", "handle_stat_unchanged",
            "path_and_handle_identity_match", "native_handle_unchanged", "bytes_match_all_recorded_sizes",
        )
        record["status"] = "verified_read" if all(record[key] for key in checks) else "mismatch_or_identity_change"
    except Exception as exc:
        record["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        record["partial_sha256"] = digest.hexdigest()
        try:
            record["path_after_error"] = stat_identity(target.stat())
        except Exception as stat_exc:
            record["errors"].append({"type": type(stat_exc).__name__, "message": str(stat_exc)})
    record["finished_at_utc"] = utc_now()
    record["diagnostic_wall_ms"] = (time.perf_counter_ns() - start) / 1e6
    dump_json(output_dir / f"pass_{ordinal:02d}.json", record)
    return record


def compare_chunks(output_dir, records):
    results = []
    baseline = records[0]
    for record in records[1:]:
        result = {"baseline_pass": baseline["pass"], "compared_pass": record["pass"], "differing_chunk_indices": []}
        try:
            with (output_dir / baseline["chunk_log"]).open(encoding="utf-8") as left, (output_dir / record["chunk_log"]).open(encoding="utf-8") as right:
                index = 0
                while True:
                    a, b = left.readline(), right.readline()
                    if not a and not b:
                        break
                    if a != b:
                        result["differing_chunk_indices"].append(index)
                    index += 1
                result["compared_chunk_slots"] = index
                result["all_recorded_chunks_equal"] = not result["differing_chunk_indices"]
        except Exception as exc:
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            result["all_recorded_chunks_equal"] = False
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--passes", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    target = args.target.resolve(strict=True)
    expected = args.expected_sha256.lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        parser.error("--expected-sha256 must contain 64 hex digits")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "started_at_utc": utc_now(), "pid": os.getpid(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "target": str(target),
        "expected_sha256": expected, "passes": args.passes,
        "chunk_bytes": CHUNK_BYTES, "source_access": "read_only_fresh_open_rb_each_serial_pass",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "performance_calibration_eligible": False,
    }
    dump_json(args.output_dir / "manifest.json", manifest)
    records = []
    for ordinal in range(1, args.passes + 1):
        record = collect_pass(target, expected, args.output_dir, ordinal)
        records.append(record)
        print(json.dumps({key: record.get(key) for key in ("pass", "pid", "status", "bytes_read", "chunk_count", "actual_sha256", "diagnostic_wall_ms")}), flush=True)
    comparisons = compare_chunks(args.output_dir, records)
    complete = all(record["status"] == "verified_read" for record in records)
    summary = {
        "schema_version": 1, "started_at_utc": manifest["started_at_utc"],
        "finished_at_utc": utc_now(), "pid": os.getpid(), "target": str(target),
        "expected_sha256": expected, "all_passes_verified": complete,
        "observed_read_failures": sum(record["status"] != "verified_read" for record in records),
        "passes": records, "chunk_comparisons": comparisons,
        "diagnostic_conclusion": (
            "No mismatch reproduced in these serial reads; historical failures remain unexplained."
            if complete else "Read mismatch or identity inconsistency observed; inspect preserved per-pass and per-chunk evidence."
        ),
        "limitations": [
            "Stable stat/handle identity and SHA during this run do not establish the cause of earlier failures.",
            "No simulator, model parser, timing gate, native evidence or GGUF file was modified.",
            "Diagnostic wall times are not inference or operator performance measurements.",
        ],
    }
    dump_json(args.output_dir / "summary.json", summary)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
