"""Observe exactly one original R33 guard call, without running lock/full/score.

The shadow reference preserves original1MiB reads and its OpenSSL reference
result. Each chunk also feeds independent _sha2 and CNG digests. No algorithm
wins a disagreement. This diagnostic never retries a failed reference or accepts
or resumes the rejected campaign. The300s budget is observational, not a kill.
"""
from contextlib import ExitStack
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
SOFT_OBSERVATION_SECONDS = 300
NAMES = ("openssl", "python_sha2", "windows_cng")
CHUNK_BYTES = 1024 * 1024


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    prior = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = prior
    return module


class DiagnosticDisagreement(ValueError):
    def __init__(self, record):
        self.record = record
        self.actual_reference = record.get("actual_reference")
        super().__init__("same-read diagnostic disagreement: " + record["path"])


class Observer:
    def __init__(self, saved, factories, core, stream):
        self.s, self.factories, self.core, self.stream = saved, factories, core, stream
        self.sequence = 0
        self.paths = {}
        self.last = None
        self.details = {}
        self.verify_count = 0
        self.source_verify_count = 0
        self.ledger_digest = factories["python_sha2"]()
        self.ledger_bytes = 0

    def preserve(self, record):
        self.details[record["sequence"]] = record

    def reference(self, path):
        self.sequence += 1
        record = {"sequence": self.sequence, "path": str(path), "started_utc": self.core.now(),
                  "read_chunk_bytes": CHUNK_BYTES, "read_length": 0, "chunk_count": 0,
                  "digests": {name: None for name in NAMES}, "algorithm_errors": {},
                  "fstat_before": None, "fstat_after": None, "path_stat_after": None,
                  "actual_reference": None, "file_open_count": 0}
        error = None
        try:
            path = Path(path).resolve(strict=True)
            record["path"] = str(path)
            self.s.require(path.is_file(), "not a file: " + str(path))
            with ExitStack() as stack:
                hashes = {}
                for name in NAMES:
                    value = self.factories[name]()
                    hashes[name] = stack.enter_context(value) if hasattr(value, "__enter__") else value
                with path.open("rb") as stream:
                    record["file_open_count"] = 1
                    record["fstat_before"] = self.core.stat_record(os.fstat(stream.fileno()))
                    for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
                        record["read_length"] += len(chunk)
                        record["chunk_count"] += 1
                        for name, digest in hashes.items():
                            if name not in record["algorithm_errors"]:
                                try:
                                    digest.update(chunk)
                                except BaseException as exc:
                                    record["algorithm_errors"][name] = self.core.error_record(exc)
                    record["fstat_after"] = self.core.stat_record(os.fstat(stream.fileno()))
                for name, digest in hashes.items():
                    if name not in record["algorithm_errors"]:
                        try:
                            record["digests"][name] = digest.hexdigest()
                        except BaseException as exc:
                            record["algorithm_errors"][name] = self.core.error_record(exc)
                # The original reference returns path.stat().st_size, not bytes-read.
                record["path_stat_after"] = self.core.stat_record(path.stat())
                record["actual_reference"] = {"path": str(path), "sha256": record["digests"]["openssl"],
                                              "bytes": record["path_stat_after"]["st_size"]}
        except BaseException as exc:
            error = exc
            record["exception"] = self.core.error_record(exc)
            record["traceback"] = traceback.format_exc()
        record["finished_utc"] = self.core.now()
        values = list(record["digests"].values())
        record["algorithms_agree"] = (not record["algorithm_errors"] and all(isinstance(v, str) for v in values)
                                       and len(set(values)) == 1)
        record["stat_unchanged"] = (record["fstat_before"] is not None and
            record["fstat_before"] == record["fstat_after"] == record["path_stat_after"])
        record["read_length_matches_path_size"] = (record["path_stat_after"] is not None and
            record["read_length"] == record["path_stat_after"]["st_size"])
        entry = self.paths.setdefault(record["path"], {"id": len(self.paths), "calls": 0, "digests": []})
        entry["calls"] += 1
        digest = record["digests"]["openssl"]
        if digest not in entry["digests"]:
            entry["digests"].append(digest)
        compact = {"i": record["sequence"], "p": entry["id"], "h": digest, "n": record["read_length"],
                   "b": (record["actual_reference"] or {}).get("bytes"), "t": record["finished_utc"]}
        encoded = (json.dumps(compact, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        self.ledger_digest.update(encoded)
        self.ledger_bytes += len(encoded)
        self.last = record
        self.stream.write(encoded.decode("utf-8"))
        self.stream.flush()
        if error is not None or not record["algorithms_agree"] or not record["stat_unchanged"] or not record["read_length_matches_path_size"]:
            self.preserve(record)
            if error is not None:
                raise error
            raise DiagnosticDisagreement(record)
        return record["actual_reference"]

    def compare(self, expected, actual, declared, mode, record=None):
        record = self.last if record is None else record
        if record is None:
            record = {"sequence": self.sequence, "path": expected["path"]}
        record["comparison"] = {"mode": mode, "declared": copy.deepcopy(declared),
            "normalized_expected": expected, "actual": actual,
            "fields": {key: actual is not None and actual.get(key) == expected.get(key) for key in ("path", "bytes", "sha256") if key in expected}}
        if not all(record["comparison"]["fields"].values()):
            self.preserve(record)
        return record

    def verify_reference(self, ref):
        self.verify_count += 1
        expected = self.s.normalized_ref(ref)
        anomaly = None
        try:
            actual = self.reference(expected["path"])
        except DiagnosticDisagreement as exc:
            actual, anomaly = exc.actual_reference, exc
        except BaseException:
            self.compare(expected, (self.last or {}).get("actual_reference"), ref, "exact_reference")
            raise
        self.compare(expected, actual, ref, "exact_reference")
        # This is precisely the original OpenSSL reference equality predicate.
        self.s.require(actual == expected, "changed evidence: " + expected["path"])
        if anomaly is not None:
            raise anomaly
        return expected

    def verify_source_evidence_reference(self, ref):
        self.source_verify_count += 1
        expected = self.s.normalized_source_evidence_ref(ref)
        compared = {"path": expected["path"], "sha256": expected["sha256"]}
        if "declared_bytes" in expected:
            compared["bytes"] = expected["declared_bytes"]
        anomaly = None
        try:
            actual = self.reference(expected["path"])
        except DiagnosticDisagreement as exc:
            actual, anomaly = exc.actual_reference, exc
        except BaseException:
            self.compare(compared, (self.last or {}).get("actual_reference"), ref, "source_evidence")
            raise
        self.compare(compared, actual, ref, "source_evidence")
        self.s.require(actual["sha256"] == expected["sha256"], "changed source evidence: " + expected["path"])
        if "declared_bytes" in expected:
            self.s.require(actual["bytes"] == expected["declared_bytes"], "changed source evidence size: " + expected["path"])
        if anomaly is not None:
            raise anomaly
        return {"path": expected["path"], "sha256": expected["sha256"], "observed_bytes": actual["bytes"],
                "declared_bytes": expected.get("declared_bytes")}

    def report(self):
        return {"reference_calls": self.sequence, "verify_reference_calls": self.verify_count,
                "source_verify_reference_calls": self.source_verify_count, "paths": self.paths,
                "ledger_bytes": self.ledger_bytes, "ledger_sha256": self.ledger_digest.hexdigest(),
                "ledger_digest_implementation": "_sha2.sha256", "full_records": list(self.details.values())}


def redirect_process_journal(function, output):
    # Same function bytecode and arguments. Only its observation-directory root
    # changes; f.P, guard inputs, Git arguments and all other globals stay intact.
    globals_copy = {**function.__globals__, "P": Path(output)}
    redirected = types.FunctionType(function.__code__, globals_copy, function.__name__,
                                    function.__defaults__, function.__closure__)
    redirected.__kwdefaults__ = function.__kwdefaults__
    return redirected


def execute(output):
    output = Path(output).resolve()
    if not output.is_relative_to(HERE) or output == HERE:
        raise ValueError("output must be a new child directory of this diagnostic folder")
    output.mkdir(parents=True, exist_ok=False)
    core = load("r33_digest_core_support", HERE / "core_digest_stability.py")
    result = {"schema": "r33-one-startup-guard-observation/v1", "started_utc": core.now(),
              "status": "incomplete", "guard_calls": 0, "pid": os.getpid(),
              "soft_observation_budget_seconds": SOFT_OBSERVATION_SECONDS, "hard_timeout": False,
              "automatic_retry": False, "native_run": False, "simulation_run": False,
              "campaign_acceptance_changed": False, "original_failure_reclassified": False,
              "algorithm_winner_selected": False, "affinity_changed": False,
              "python_executable": sys.executable, "python_version": sys.version,
              "shadowed_functions": ["s.reference", "s.verify_reference", "s.verify_source_evidence_reference"],
              "original_guard_data_root": str(ROUND),
              "process_journal_directory": str(output / "process_observations"),
              "process_journal_function_code_preserved": True,
              "diagnostic_extra_fail_closed": ["same-read algorithm disagreement", "read-time file identity or length instability"]}
    observer = None
    started = time.monotonic()
    try:
        factories, helper, helper_bytes, helper_read, algorithms = core.load_factories()
        result["algorithms"] = algorithms
        result["cng_helper_read"] = helper_read
        result["known_vectors"] = core.known_vectors(factories)
        if any(not all(row["expected_matches"].values()) for row in result["known_vectors"]):
            raise ValueError("known vector failed; original guard not called")
        runner = load("r33_original_guard_observed", ROUND / "run_candidate.py")
        originals = {name: getattr(runner.s, name) for name in ("reference", "verify_reference", "verify_source_evidence_reference")}
        original_process = runner.f.natural_process
        with (output / "references.jsonl").open("x", encoding="utf-8") as stream:
            observer = Observer(runner.s, factories, core, stream)
            try:
                for name in originals:
                    setattr(runner.s, name, getattr(observer, name))
                runner.f.natural_process = redirect_process_journal(original_process, output)
                result["source_refs"] = {name: observer.reference(path) for name, path in {
                    "observer": __file__, "core_support": HERE / "core_digest_stability.py", "cng_helper": core.HELPER,
                    "runner": runner.__file__, "freezer": runner.f.__file__, "evidence_helper": runner.s.__file__}.items()}
                core.write_new(output / "start.json", result)
                result["guard_calls"] = 1
                result["guard_result"] = runner.guard()
                result["status"] = "diagnostic_guard_returned"
            finally:
                for name, function in originals.items():
                    setattr(runner.s, name, function)
                runner.f.natural_process = original_process
    except BaseException as exc:
        result["status"] = "diagnostic_guard_failed" if result["guard_calls"] else "diagnostic_setup_failed"
        result["error"] = core.error_record(exc)
        result["traceback"] = traceback.format_exc()
    if observer is not None:
        result["observation"] = observer.report()
    result["elapsed_seconds"] = time.monotonic() - started
    result["soft_observation_budget_exceeded"] = result["elapsed_seconds"] > SOFT_OBSERVATION_SECONDS
    result["finished_utc"] = core.now()
    core.write_new(output / "finish.json", result)
    print(json.dumps({key: result.get(key) for key in ("status", "guard_calls", "elapsed_seconds", "error")}))
    return 0 if result["status"] == "diagnostic_guard_returned" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observe-once", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.observe_once:
        parser.error("explicit --observe-once required")
    return execute(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
