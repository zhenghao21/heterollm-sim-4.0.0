"""Read-only identity snapshots of modules actually loaded by a process.

This is runtime provenance, not a proof of counter semantics, performance
compatibility, or in-memory code identity.  There is deliberately no directory
scan fallback.  ``artifacts`` is EXE-first for callers using the existing native
artifact convention; ``current_modules`` retains loader metadata too.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone
from typing import Any, Mapping


SCHEMA = "native-loaded-runtime/v1"


def _path_key(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _file_ref(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        size = 0
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(handle.fileno())
    fingerprint = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if fingerprint(before) != fingerprint(after) or fingerprint(after) != fingerprint(path.stat()):
        raise RuntimeError(f"loaded module changed while hashing: {path}")
    if size <= 0 or size != after.st_size:
        raise RuntimeError(f"loaded module is empty or incomplete: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": size,
            "status": "captured", "file_mtime_ns": after.st_mtime_ns}


def _capture_windows_psapi(pid: int) -> dict[str, object]:
    """Enumerate a live loader list using PSAPI, never a filesystem glob.

    A 64-bit collector is required: 32-bit WOW64 enumeration cannot guarantee a
    complete 64-bit target list.  PSAPI excludes LOAD_LIBRARY_AS_DATAFILE images.
    """
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("64-bit Windows module collector required")
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    psapi.EnumProcessModulesEx.argtypes = (
        wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), wintypes.DWORD,
    )
    psapi.EnumProcessModulesEx.restype = wintypes.BOOL
    psapi.GetModuleFileNameExW.argtypes = (
        wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD,
    )
    psapi.GetModuleFileNameExW.restype = wintypes.DWORD

    class MODULEINFO(ctypes.Structure):
        _fields_ = (("base", wintypes.LPVOID), ("size", wintypes.DWORD),
                    ("entry", wintypes.LPVOID))

    psapi.GetModuleInformation.argtypes = (
        wintypes.HANDLE, wintypes.HMODULE, ctypes.POINTER(MODULEINFO), wintypes.DWORD,
    )
    psapi.GetModuleInformation.restype = wintypes.BOOL

    def failed(api: str) -> OSError:
        error = ctypes.get_last_error()
        return OSError(error, f"{api} failed for pid={pid}: {ctypes.FormatError(error).strip()}")

    handle = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
    if not handle:
        raise failed("OpenProcess")
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise failed("GetExitCodeProcess")
        if exit_code.value != 259:  # STILL_ACTIVE
            raise ProcessLookupError(f"process {pid} has exited")
        creation, exit_time, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
        if not kernel32.GetProcessTimes(handle, *map(ctypes.byref, (creation, exit_time, kernel_time, user_time))):
            raise failed("GetProcessTimes")
        creation_ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        capacity = 128
        handles: Any = None
        count = 0
        for _ in range(10):
            handles = (wintypes.HMODULE * capacity)()
            needed = wintypes.DWORD()
            if not psapi.EnumProcessModulesEx(handle, handles, ctypes.sizeof(handles), ctypes.byref(needed), 0x03):
                raise failed("EnumProcessModulesEx")
            width = ctypes.sizeof(wintypes.HMODULE)
            if needed.value % width:
                raise RuntimeError("module enumeration returned a partial handle")
            count = needed.value // width
            if count <= capacity:
                break
            capacity = count + 32
        else:
            raise RuntimeError("module list did not stabilize during enumeration")
        if not count:
            raise RuntimeError("module enumeration returned an empty list")
        modules = []
        for module in handles[:count]:
            capacity_chars = 1024
            while capacity_chars <= 65536:
                filename = ctypes.create_unicode_buffer(capacity_chars)
                length = psapi.GetModuleFileNameExW(handle, module, filename, capacity_chars)
                if not length:
                    raise failed("GetModuleFileNameExW")
                if length < capacity_chars - 1:
                    break
                capacity_chars *= 2
            else:
                raise RuntimeError("module path exceeds collection limit")
            info = MODULEINFO()
            if not psapi.GetModuleInformation(handle, module, ctypes.byref(info), ctypes.sizeof(info)):
                raise failed("GetModuleInformation")
            modules.append({"path": filename.value, "base_address": int(info.base or 0),
                            "image_size_bytes": int(info.size)})
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise failed("GetExitCodeProcess")
        if exit_code.value != 259:
            raise ProcessLookupError(f"process {pid} exited during enumeration")
        return {"method": "windows_psapi", "process_identity": str(creation_ticks),
                "modules": modules}
    finally:
        kernel32.CloseHandle(handle)


def _parse_linux_maps(text: str) -> list[dict[str, object]]:
    """Select file-backed executable mappings, excluding model mmap buffers."""
    segments: dict[str, list[tuple[int, int, str]]] = {}
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        start_hex, end_hex = fields[0].split("-", 1)
        start, end = int(start_hex, 16), int(end_hex, 16)
        if end <= start:
            raise ValueError("invalid process map address range")
        path = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[5])
        segments.setdefault(path, []).append((start, end, fields[1]))
    modules = []
    for path, ranges in segments.items():
        if not any("x" in permissions for _, _, permissions in ranges):
            continue
        if path.endswith(" (deleted)"):
            raise RuntimeError(f"loaded module no longer has verifiable file bytes: {path}")
        modules.append({"path": path, "base_address": min(start for start, _, _ in ranges),
                        "image_size_bytes": sum(end - start for start, end, _ in ranges)})
    return modules


def _capture_linux_proc_maps(pid: int) -> dict[str, object]:
    proc = Path("/proc") / str(pid)
    stat = (proc / "stat").read_text(encoding="utf-8")
    # comm may contain spaces and parentheses; fields after the last ')' start
    # at field 3, hence index 19 is starttime (field 22).
    identity_fields = stat[stat.rfind(")") + 2:].split()
    if len(identity_fields) <= 19 or identity_fields[0] == "Z":
        raise ProcessLookupError(f"process {pid} is absent, exited, or malformed")
    starttime = identity_fields[19]
    modules = _parse_linux_maps((proc / "maps").read_text(encoding="utf-8"))
    if not modules:
        raise RuntimeError("process has no verifiable executable mappings")
    return {"method": "linux_proc_maps", "process_identity": starttime, "modules": modules}


def _capture_platform_modules(pid: int) -> dict[str, object]:
    if sys.platform == "win32":
        return _capture_windows_psapi(pid)
    if sys.platform.startswith("linux"):
        return _capture_linux_proc_maps(pid)
    raise RuntimeError(f"loaded runtime collection is unsupported on {sys.platform}")


def _module_signature(snapshot: Mapping[str, object]) -> tuple[tuple[str, object, object], ...]:
    modules = snapshot.get("modules")
    if not isinstance(modules, list) or not modules:
        raise RuntimeError("actual loaded module set is missing or empty")
    if not snapshot.get("process_identity") or not snapshot.get("method"):
        raise RuntimeError("process identity or enumeration method missing")
    rows = []
    for module in modules:
        if not isinstance(module, Mapping) or not isinstance(module.get("path"), str) or not module["path"].strip():
            raise RuntimeError("actual module path is missing or malformed")
        rows.append((_path_key(module["path"]), module.get("base_address"), module.get("image_size_bytes")))
    return tuple(sorted(rows, key=lambda row: (row[0], str(row[1]), str(row[2]))))


def capture_loaded_runtime(pid: int, exe: str | Path) -> dict[str, object]:
    """Capture a stable live module set and file identities; fail closed.

    The process creation identity and module list are observed before and after
    hashing.  Failure leaves ``artifacts``/``current_modules`` empty so partial
    evidence cannot silently become a complete native runtime identity.
    """
    result: dict[str, object] = {
        "schema": SCHEMA, "status": "failed", "pid": pid,
        "exe": None, "method": None,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": [], "current_modules": [], "actual_modules": [],
        "module_sha256_set": [], "errors": [],
        "limitations": [
            "Snapshot covers loader-visible modules at collection time, not later lazy loads.",
            "File SHA identifies on-disk module bytes, not executable memory or generated GPU code.",
            "Windows PSAPI excludes images loaded only as data files; Linux uses executable file mappings.",
        ],
    }
    try:
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("pid must be a positive integer")
        if not isinstance(exe, (str, Path)) or not str(exe).strip():
            raise ValueError("exe must name the expected executable")
        expected_exe = Path(exe).resolve(strict=True)
        result["exe"] = str(expected_exe)
        first = _capture_platform_modules(pid)
        result["method"] = first.get("method")
        initial_signature = _module_signature(first)
        expected_key = _path_key(expected_exe)
        if expected_key not in {row[0] for row in initial_signature}:
            raise RuntimeError("expected executable is not in the actual loaded module set")
        # Deduplicate file identities while retaining separate mapping metadata.
        refs: dict[str, dict[str, object]] = {}
        for path, _, _ in initial_signature:
            if path not in refs:
                refs[path] = _file_ref(Path(path))
        second = _capture_platform_modules(pid)
        if (first.get("method") != second.get("method") or
                first.get("process_identity") != second.get("process_identity") or
                initial_signature != _module_signature(second)):
            raise RuntimeError("process identity or loaded module set changed during capture")
        for path, ref in refs.items():
            stat = Path(path).stat()
            if stat.st_size != ref["bytes"] or stat.st_mtime_ns != ref["file_mtime_ns"]:
                raise RuntimeError(f"loaded module file changed during capture: {path}")
        keys = sorted(refs, key=lambda key: (key != expected_key, key))
        artifacts = [dict(refs[key], role="executable" if key == expected_key else "loaded_module") for key in keys]
        current_modules = [dict(refs[path], base_address=base, image_size_bytes=image_size,
                                role="executable" if path == expected_key else "loaded_module")
                           for path, base, image_size in sorted(initial_signature, key=lambda row: (row[0] != expected_key, row[0], str(row[1])))]
        identity = [{"path": row["path"], "sha256": row["sha256"], "bytes": row["bytes"]} for row in artifacts]
        identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        result.update(status="captured", process_identity=first["process_identity"],
                      artifacts=artifacts, current_modules=current_modules, actual_modules=current_modules,
                      module_count=len(current_modules), module_identity_sha256=identity_sha,
                      module_sha256_set=sorted({str(row["sha256"]) for row in artifacts}),
                      completed_at=datetime.now(timezone.utc).isoformat())
    except (OSError, ValueError, RuntimeError, TypeError, AttributeError, OverflowError) as error:
        result["errors"] = [f"{type(error).__name__}: {error}"]
    return result


__all__ = ["SCHEMA", "capture_loaded_runtime"]