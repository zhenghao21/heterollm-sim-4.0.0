"""Bounded SHA256 stability diagnosis; never native, simulation, retry, or calibration.

Execute only after review, in this separate diagnostic process. Read the fixed
PDF once, share immutable 1MiB blocks, visit the initial process affinity cores
in increasing order, and run exactly16 trials per started core. A300s soft
budget and observer errors may prevent starting another core; neither interrupts
an already started core. Restore this process affinity after every core.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import ctypes as C
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import ssl
import sys
import time
import types

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
HELPER = ROUND.parent / "round_027/identity_diagnostic/dual_hash.py"
PDF = ROUND / "off/evidence/nvidia-rtx-blackwell-gpu-architecture.pdf"
EXPECTED_SIZE = 8283392
EXPECTED_SHA256 = "906ff2a409d7a7e4cbc56f5d3a179d574120d19aaba99520670e1a0c064595fa"
CHUNK_BYTES = 1024 * 1024
TRIALS_PER_CORE = 16
SOFT_BUDGET_SECONDS = 300.0
ALGORITHMS = ("openssl", "python_sha2", "windows_cng")
IDENTITY_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
VECTORS = (
    ("empty", b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ("abc", b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
    ("million_a", b"a" * 1000000, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def error_record(exc):
    return {"type": type(exc).__name__, "message": str(exc),
            "notes": list(getattr(exc, "__notes__", [])),
            "cng_cleanup_errors": list(getattr(exc, "cng_cleanup_errors", []))}


def stat_record(stat):
    return {name: getattr(stat, name) for name in IDENTITY_FIELDS}


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def blocks(payload):
    if type(payload) is not bytes:
        raise TypeError("immutable bytes required")
    return tuple(payload[offset:offset + CHUNK_BYTES] for offset in range(0, len(payload), CHUNK_BYTES))


def digest_blocks(factory, chunks):
    with ExitStack() as stack:
        digest = factory()
        if hasattr(digest, "__enter__"):
            digest = stack.enter_context(digest)
        for chunk in chunks:
            digest.update(chunk)
        return digest.hexdigest()


def digest_observation(chunks, factories, expected):
    result = {"started_utc": now(), "digests": {}, "errors": {}}
    for name in ALGORITHMS:
        try:
            result["digests"][name] = digest_blocks(factories[name], chunks)
        except BaseException as exc:
            result["digests"][name] = None
            result["errors"][name] = error_record(exc)
    result["expected_matches"] = {name: value == expected for name, value in result["digests"].items()}
    values = list(result["digests"].values())
    result["algorithms_agree"] = not result["errors"] and len(set(values)) == 1
    result["expected_sha256"] = expected
    result["finished_utc"] = now()
    return result


def known_vectors(factories):
    return [{"name": name, "bytes": len(payload), **digest_observation(blocks(payload), factories, expected)}
            for name, payload, expected in VECTORS]


def read_once(path):
    """Exactly one open/read; never retry a failed or changed read."""
    path = Path(path).resolve(strict=True)
    record = {"path": str(path), "started_utc": now(), "file_open_count": 1,
              "read_calls": 1, "extra_eof_probe": False}
    with path.open("rb") as stream:
        record["fstat_before"] = stat_record(os.fstat(stream.fileno()))
        payload = stream.read()
        record["fstat_after"] = stat_record(os.fstat(stream.fileno()))
    try:
        record["path_stat_after"] = stat_record(path.stat())
    except Exception as exc:
        record["path_stat_after"] = None
        record["path_stat_error"] = error_record(exc)
    record["actual_bytes"] = len(payload)
    record["identity_unchanged"] = record["fstat_before"] == record["fstat_after"] == record["path_stat_after"]
    record["finished_utc"] = now()
    return payload, record


def load_factories():
    import _hashlib
    import _sha2
    helper_bytes, helper_read = read_once(HELPER)
    helper = types.ModuleType("r33_fixed_cng_diagnostic")
    helper.__file__ = str(HELPER)
    exec(compile(helper_bytes, str(HELPER), "exec"), helper.__dict__)
    factories = {"openssl": _hashlib.openssl_sha256, "python_sha2": _sha2.sha256,
                 "windows_cng": helper.CNGSHA256}
    if type(factories["openssl"]()).__module__ != "_hashlib" or type(factories["python_sha2"]()).__module__ != "_sha2":
        raise RuntimeError("required independent implementation missing; no fallback")
    return factories, helper, helper_bytes, helper_read, {
        "openssl_constructor": "_hashlib.openssl_sha256", "openssl_module": _hashlib.__file__,
        "openssl_version": ssl.OPENSSL_VERSION, "python_sha2_constructor": "_sha2.sha256",
        "python_sha2_origin": importlib.util.find_spec("_sha2").origin,
        "cng_constructor": "round_027.identity_diagnostic.dual_hash.CNGSHA256",
        "cng_provider": helper.PROVIDER, "fallback_used": False}


class CurrentProcessAffinity:
    """Only the current diagnostic process; one Windows processor group."""
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("Windows affinity required; no substitute")
        self.dll = C.WinDLL("kernel32", use_last_error=True)
        for name, args, restype in (
            ("GetCurrentProcess", [], C.c_void_p),
            ("GetProcessAffinityMask", [C.c_void_p, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)], C.c_int),
            ("SetProcessAffinityMask", [C.c_void_p, C.c_size_t], C.c_int),
            ("GetCurrentProcessorNumber", [], C.c_uint32),
            ("GetActiveProcessorGroupCount", [], C.c_uint16),
            ("GetModuleFileNameW", [C.c_void_p, C.c_wchar_p, C.c_uint32], C.c_uint32),
        ):
            fn = getattr(self.dll, name)
            fn.argtypes, fn.restype = args, restype
        if self.dll.GetActiveProcessorGroupCount() != 1:
            raise RuntimeError("multiple processor groups require a separately reviewed diagnostic")
        self.handle = self.dll.GetCurrentProcess()
        first = self.observe()
        self.original_mask = first["process_mask"]
        self.system_mask = first["system_mask"]
        if not self.original_mask or self.original_mask & ~self.system_mask:
            raise RuntimeError("invalid original process affinity")
        self.cores = [core for core in range(C.sizeof(C.c_size_t) * 8) if self.original_mask & (1 << core)]

    def observe(self):
        process_mask, system_mask = C.c_size_t(), C.c_size_t()
        if not self.dll.GetProcessAffinityMask(self.handle, C.byref(process_mask), C.byref(system_mask)):
            raise C.WinError(C.get_last_error())
        return {"process_mask": process_mask.value, "system_mask": system_mask.value,
                "current_cpu": self.dll.GetCurrentProcessorNumber(), "observed_utc": now()}

    def set_mask(self, mask):
        if not self.dll.SetProcessAffinityMask(self.handle, mask):
            raise C.WinError(C.get_last_error())
        actual = self.observe()
        if actual["process_mask"] != mask:
            raise RuntimeError("requested process affinity was not observed")
        return actual

    @contextmanager
    def pinned(self, core):
        if core not in self.cores:
            raise ValueError("core outside the original process affinity")
        try:
            self.set_mask(1 << core)
            yield
        finally:
            self.set_mask(self.original_mask)

    def python_dll_path(self):
        buffer = C.create_unicode_buffer(32768)
        size = self.dll.GetModuleFileNameW(C.pythonapi._handle, buffer, len(buffer))
        if not size or size >= len(buffer):
            raise RuntimeError("cannot identify the loaded Python runtime DLL")
        return buffer.value


def runtime_identity(factories, helper, helper_bytes, helper_read, algorithms, affinity):
    paths = {"script": str(Path(__file__).resolve()), "python_executable": sys.executable,
             "python_runtime_dll": affinity.python_dll_path(), "openssl_extension": algorithms["openssl_module"],
             "cng_dll": str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/bcrypt.dll")}
    helper_digest = digest_observation(blocks(helper_bytes), factories, None)
    files = {"cng_helper": {**helper_read, "digests": helper_digest["digests"],
             "errors": helper_digest["errors"], "algorithms_agree": helper_digest["algorithms_agree"]}}
    for name, path in paths.items():
        payload, read = read_once(path)
        observed = digest_observation(blocks(payload), factories, None)
        files[name] = {**read, "digests": observed["digests"], "errors": observed["errors"],
                       "algorithms_agree": observed["algorithms_agree"]}
    return {"pid": os.getpid(), "executable": sys.executable, "version": sys.version,
            "implementation": platform.python_implementation(), "machine": platform.machine(),
            "platform": sys.platform, "algorithms": algorithms, "files": files,
            "identity_scope": "listed runtime/source files only; not all transitive DLL dependencies"}


def run_schedule(chunks, expected, factories, affinity, emit, *, clock=time.monotonic,
                 started=None, trials_per_core=TRIALS_PER_CORE, soft_budget=SOFT_BUDGET_SECONDS):
    """Finite schedule. Observer failure waits for the current core, then starts no more."""
    if type(chunks) is not tuple or any(type(chunk) is not bytes or len(chunk) > CHUNK_BYTES for chunk in chunks):
        raise TypeError("a fixed tuple of immutable <=1MiB chunks is required")
    if tuple(factories) != ALGORITHMS or trials_per_core < 1:
        raise ValueError("fixed algorithm order and positive trial count required")
    started = clock() if started is None else started
    records, observer_errors, core_errors, completed = [], [], [], []
    stop_reason = None
    for core in affinity.cores:
        if clock() - started >= soft_budget:
            stop_reason = "soft_budget_before_next_core"
            break
        core_record = {"core": core, "mask": hex(1 << core), "started_utc": now(), "trials": []}
        records.append(core_record)
        try:
            with affinity.pinned(core):
                core_record["known_vectors"] = known_vectors(factories)
                for iteration in range(trials_per_core):
                    row = {"kind": "pdf_trial", "core": core, "mask": hex(1 << core),
                           "iteration": iteration, "input_bytes": sum(map(len, chunks)), "chunk_bytes": CHUNK_BYTES}
                    try:
                        row["affinity_before"] = affinity.observe()
                    except BaseException as exc:
                        row["affinity_before_error"] = error_record(exc)
                        observer_errors.append({"core": core, "iteration": iteration, "stage": "affinity_before", **error_record(exc)})
                    row.update(digest_observation(chunks, factories, expected))
                    try:
                        row["affinity_after"] = affinity.observe()
                    except BaseException as exc:
                        row["affinity_after_error"] = error_record(exc)
                        observer_errors.append({"core": core, "iteration": iteration, "stage": "affinity_after", **error_record(exc)})
                    row["affinity_verified"] = all(row.get(key, {}).get("process_mask") == 1 << core
                        and row.get(key, {}).get("current_cpu") == core for key in ("affinity_before", "affinity_after"))
                    core_record["trials"].append(row)
                    try:
                        emit(row)
                    except BaseException as exc:
                        observer_errors.append({"core": core, "iteration": iteration, "stage": "journal", **error_record(exc)})
            core_record["affinity_after_restore"] = affinity.observe()
            core_record["restored"] = core_record["affinity_after_restore"]["process_mask"] == affinity.original_mask
            if not core_record["restored"]:
                raise RuntimeError("original diagnostic process affinity not restored")
            completed.append(core)
        except BaseException as exc:
            core_record["error"] = error_record(exc)
            core_errors.append({"core": core, **error_record(exc)})
            stop_reason = "core_control_failure_no_new_core"
        core_record["finished_utc"] = now()
        if core_errors or observer_errors:
            stop_reason = stop_reason or "observer_failure_after_completed_core"
            break
    remaining = [core for core in affinity.cores if core not in [record["core"] for record in records]]
    trials = [row for record in records for row in record["trials"]]
    return {"core_records": records, "completed_cores": completed,
            "unstarted_cores": [{"core": core, "reason": stop_reason} for core in remaining],
            "planned_trials": len(affinity.cores) * trials_per_core, "completed_trials": len(trials),
            "observer_errors": observer_errors, "core_errors": core_errors, "stop_reason": stop_reason,
            "all_planned_completed": len(completed) == len(affinity.cores),
            "mismatch_trials": sum(not all(row["expected_matches"].values()) for row in trials),
            "algorithm_disagreement_trials": sum(not row["algorithms_agree"] for row in trials),
            "affinity_unverified_trials": sum(not row["affinity_verified"] for row in trials),
            "known_vector_failures": sum(not all(row["expected_matches"].values())
                for record in records for row in record.get("known_vectors", []))}


def execute(output):
    output = Path(output).resolve()
    if not output.is_relative_to(HERE) or output == HERE:
        raise ValueError("diagnostic output must be a new child directory of this script")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"schema": "r33-core-sha256-stability/v1", "started_utc": now(), "status": "incomplete",
              "read_only_source": True, "native_run": False, "simulation_run": False,
              "target_latency_used": False, "calibration_parameters": 0, "automatic_retry": False,
              "original_failure_reclassified": False, "algorithm_winner_selected": False,
              "process_scope": "only this separately launched diagnostic process", "pid": os.getpid()}
    try:
        factories, helper, helper_bytes, helper_read, algorithms = load_factories()
        affinity = CurrentProcessAffinity()
        result["runtime_identity"] = runtime_identity(factories, helper, helper_bytes, helper_read, algorithms, affinity)
        result["runtime_identity_disagreements"] = [name for name, ref in result["runtime_identity"]["files"].items()
            if not ref["identity_unchanged"] or not ref["algorithms_agree"]]
        payload, input_read = read_once(PDF)
        chunks = blocks(payload)
        result["input"] = {**input_read, "expected_bytes": EXPECTED_SIZE, "expected_sha256": EXPECTED_SHA256,
                           "size_matches": len(payload) == EXPECTED_SIZE, "chunk_lengths": list(map(len, chunks)),
                           "reuse_policy": "one read; precomputed tuple of identical immutable bytes objects for all implementations and trials"}
        result["plan"] = {"available_cores": affinity.cores, "original_process_mask": hex(affinity.original_mask),
                          "system_mask": hex(affinity.system_mask), "trials_per_core": TRIALS_PER_CORE,
                          "algorithms": list(ALGORITHMS), "soft_budget_seconds": SOFT_BUDGET_SECONDS,
                          "hard_timeout": False, "budget_boundary": "only before starting a core"}
        write_new(output / "start.json", result)
        with (output / "trials.jsonl").open("x", encoding="utf-8") as stream:
            def emit(row):
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
            result.update(run_schedule(chunks, EXPECTED_SHA256, factories, affinity, emit, started=started))
        if result["all_planned_completed"] and not result["observer_errors"] and not result["core_errors"]:
            disagreed = any(result[name] for name in ("mismatch_trials", "algorithm_disagreement_trials",
                "affinity_unverified_trials", "known_vector_failures", "runtime_identity_disagreements"))
            disagreed = disagreed or not input_read["identity_unchanged"] or len(payload) != EXPECTED_SIZE
            result["status"] = "completed_with_disagreements" if disagreed else "completed_without_observed_disagreement"
    except BaseException as exc:
        result["error"] = error_record(exc)
    result["finished_utc"] = now()
    write_new(output / "finish.json", result)
    print(json.dumps({key: result.get(key) for key in ("status", "planned_trials", "completed_trials", "mismatch_trials", "stop_reason", "error")}))
    return 0 if result["status"].startswith("completed_") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-affinity-diagnostic", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.execute_affinity_diagnostic or args.output is None:
        parser.error("reviewed execution requires --execute-affinity-diagnostic and a new --output directory")
    return execute(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
