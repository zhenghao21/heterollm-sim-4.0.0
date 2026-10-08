"""Check access to a dedicated ETW session without recording application events.

This is a permission probe, not a GPU benchmark. It never requests elevation,
changes group membership, starts WPR's broad GPU profile, or writes an ETL.
The provider is enabled only with an allow-list for unused event ID 65535;
no event consumer is attached, and the dedicated session is immediately stopped.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid


class GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]


class WNODE_HEADER(ctypes.Structure):
    _fields_ = [("BufferSize", wintypes.ULONG), ("ProviderId", wintypes.ULONG),
                ("HistoricalContext", ctypes.c_ulonglong), ("TimeStamp", ctypes.c_longlong),
                ("Guid", GUID), ("ClientContext", wintypes.ULONG), ("Flags", wintypes.ULONG)]


class EVENT_TRACE_PROPERTIES(ctypes.Structure):
    _fields_ = [("Wnode", WNODE_HEADER)] + [(name, wintypes.ULONG) for name in
        ("BufferSize", "MinimumBuffers", "MaximumBuffers", "MaximumFileSize", "LogFileMode",
         "FlushTimer", "EnableFlags", "AgeLimit", "NumberOfBuffers", "FreeBuffers", "EventsLost",
         "BuffersWritten", "LogBuffersLost", "RealTimeBuffersLost")] + [
        ("LoggerThreadId", wintypes.HANDLE), ("LogFileNameOffset", wintypes.ULONG),
        ("LoggerNameOffset", wintypes.ULONG)]


class EVENT_FILTER_DESCRIPTOR(ctypes.Structure):
    _fields_ = [("Ptr", ctypes.c_ulonglong), ("Size", wintypes.ULONG), ("Type", wintypes.ULONG)]


class EVENT_FILTER_EVENT_ID(ctypes.Structure):
    _fields_ = [("FilterIn", ctypes.c_ubyte), ("Reserved", ctypes.c_ubyte),
                ("Count", wintypes.USHORT), ("Events", wintypes.USHORT * 1)]


class ENABLE_TRACE_PARAMETERS(ctypes.Structure):
    _fields_ = [("Version", wintypes.ULONG), ("EnableProperty", wintypes.ULONG),
                ("ControlFlags", wintypes.ULONG), ("SourceId", GUID),
                ("EnableFilterDesc", ctypes.POINTER(EVENT_FILTER_DESCRIPTOR)),
                ("FilterDescCount", wintypes.ULONG)]


def check_access():
    if os.name != "nt":
        raise RuntimeError("Windows ETW access check requires Windows")
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    start = advapi.StartTraceW
    start.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), wintypes.LPCWSTR, ctypes.POINTER(EVENT_TRACE_PROPERTIES)]
    start.restype = wintypes.ULONG
    control = advapi.ControlTraceW
    control.argtypes = [ctypes.c_ulonglong, wintypes.LPCWSTR, ctypes.POINTER(EVENT_TRACE_PROPERTIES), wintypes.ULONG]
    control.restype = wintypes.ULONG
    enable = advapi.EnableTraceEx2
    enable.argtypes = [ctypes.c_ulonglong, ctypes.POINTER(GUID), wintypes.ULONG, ctypes.c_ubyte,
                       ctypes.c_ulonglong, ctypes.c_ulonglong, wintypes.ULONG, ctypes.POINTER(ENABLE_TRACE_PARAMETERS)]
    enable.restype = wintypes.ULONG
    name = f"HeteroLLMSim-DispatchAccess-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    encoded = (name + "\0").encode("utf-16-le")
    storage = ctypes.create_string_buffer(ctypes.sizeof(EVENT_TRACE_PROPERTIES) + len(encoded))
    properties = EVENT_TRACE_PROPERTIES.from_buffer(storage)
    properties.Wnode.BufferSize = len(storage)
    properties.Wnode.ClientContext = 1  # QueryPerformanceCounter clock.
    properties.Wnode.Flags = 0x00020000  # WNODE_FLAG_TRACED_GUID.
    properties.BufferSize = 64
    properties.MinimumBuffers = 2
    properties.MaximumBuffers = 4
    properties.LogFileMode = 0x100 | 0x10000000  # REAL_TIME, NO_PER_PROCESSOR_BUFFERING.
    properties.FlushTimer = 1
    properties.LoggerNameOffset = ctypes.sizeof(EVENT_TRACE_PROPERTIES)
    ctypes.memmove(ctypes.addressof(storage) + properties.LoggerNameOffset, encoded, len(encoded))
    handle = ctypes.c_ulonglong()
    result = {"schema": "heterollm.cuda-dispatch-etw-access/v1", "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "dedicated_memory_session_access_check_no_application_event_collection",
              "provider": "Microsoft-Windows-DxgKrnl", "event_id_allowlist": [65535],
              "etl_written": False, "consumer_started": False, "elevation_requested": False,
              "start_trace_status": int(start(ctypes.byref(handle), name, ctypes.byref(properties)))}
    if result["start_trace_status"] != 0:
        result["start_trace_error"] = ctypes.FormatError(result["start_trace_status"]).strip()
        result["provider_enable_attempted"] = False
        result["session_started"] = False
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        return result
    result["session_started"] = True
    try:
        provider = GUID.from_buffer_copy(uuid.UUID("802ec45a-1e99-4b83-9920-87c98277ba9d").bytes_le)
        event_filter = EVENT_FILTER_EVENT_ID(1, 0, 1, (wintypes.USHORT * 1)(65535))
        descriptor = EVENT_FILTER_DESCRIPTOR(ctypes.addressof(event_filter), ctypes.sizeof(event_filter), 0x80000200)
        params = ENABLE_TRACE_PARAMETERS()
        params.Version = 2
        params.EnableFilterDesc = ctypes.pointer(descriptor)
        params.FilterDescCount = 1
        status = int(enable(handle, ctypes.byref(provider), 1, 4, 0x04008041, 0, 0, ctypes.byref(params)))
        result["provider_enable_attempted"] = True
        result["enable_trace_status"] = status
        if status:
            result["enable_trace_error"] = ctypes.FormatError(status).strip()
    finally:
        result["stop_trace_status"] = int(control(handle, name, ctypes.byref(properties), 1))
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_access()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result.get("stop_trace_status", 0):
        raise RuntimeError("dedicated ETW session could not be stopped")


if __name__ == "__main__":
    main()
