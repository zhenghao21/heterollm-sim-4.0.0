"""Synthetic data and fake affinity only; never opens the campaign PDF or models."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("core_digest_stability.py")
spec = importlib.util.spec_from_file_location("r33_core_digest_test", SCRIPT)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class Affinity:
    def __init__(self, cores=(0, 2), *, observe_error=None, restore_error=False):
        self.cores = list(cores)
        self.original_mask = self.system_mask = sum(1 << core for core in cores)
        self.mask = self.original_mask
        self.current_cpu = cores[0]
        self.visits = []
        self.restores = []
        self.observe_count = 0
        self.observe_error = observe_error
        self.restore_error = restore_error
    def observe(self):
        self.observe_count += 1
        if self.observe_count == self.observe_error:
            raise OSError("synthetic observation failure")
        return {"process_mask": self.mask, "system_mask": self.system_mask, "current_cpu": self.current_cpu}
    @contextmanager
    def pinned(self, core):
        self.visits.append(core)
        self.mask, self.current_cpu = 1 << core, core
        try:
            yield
        finally:
            self.mask = self.original_mask
            self.restores.append(self.mask)
            if self.restore_error:
                raise OSError("synthetic restore verification failure")


def factories(chunks, *, wrong=None, raises=None, seen=None, bad_vector=False):
    target_ids = {id(chunk) for chunk in chunks}
    def factory(name):
        class Hash:
            def __init__(self):
                self.hash = hashlib.sha256()
                self.is_target = False
            def update(self, chunk):
                if id(chunk) in target_ids:
                    self.is_target = True
                    if seen is not None:
                        seen.append((name, id(chunk)))
                    if name == raises:
                        raise RuntimeError("synthetic hash error")
                self.hash.update(chunk)
            def hexdigest(self):
                if (self.is_target and name == wrong) or (bad_vector and not self.is_target and name == wrong):
                    return "0" * 64
                return self.hash.hexdigest()
        return Hash
    return {name: factory(name) for name in d.ALGORITHMS}


def trial_input():
    payload = b"synthetic PDF input, no real evidence read"
    return d.blocks(payload), hashlib.sha256(payload).hexdigest()


def run(affinity=None, **kwargs):
    chunks, expected = trial_input()
    affinity = affinity or Affinity()
    return d.run_schedule(chunks, expected, factories(chunks), affinity, lambda row: None, **kwargs)


def test_one_file_open_and_logical_read_without_eof_probe(tmp_path, monkeypatch):
    path = tmp_path / "input.bin"
    payload = b"synthetic input"
    path.write_bytes(payload)
    original = Path.open
    counts = {"opens": 0, "reads": []}
    class Stream:
        def __init__(self, inner): self.inner = inner
        def __enter__(self): return self
        def __exit__(self, *args): return self.inner.__exit__(*args)
        def fileno(self): return self.inner.fileno()
        def read(self, *args):
            counts["reads"].append(args)
            return self.inner.read(*args)
    def opened(self, *args, **kwargs):
        inner = original(self, *args, **kwargs)
        if self == path:
            counts["opens"] += 1
            return Stream(inner)
        return inner
    monkeypatch.setattr(Path, "open", opened)
    value, record = d.read_once(path)
    assert value == payload and record["identity_unchanged"]
    assert counts == {"opens": 1, "reads": [()]}
    assert record["file_open_count"] == record["read_calls"] == 1


def test_all_cores_exact16_trials_same_chunk_objects_restore_each_core():
    chunks, expected = trial_input()
    seen, emitted = [], []
    affinity = Affinity()
    result = d.run_schedule(chunks, expected, factories(chunks, seen=seen), affinity, emitted.append)
    assert result["all_planned_completed"] and result["planned_trials"] == result["completed_trials"] == 32
    assert affinity.visits == [0, 2] and affinity.restores == [5, 5] and affinity.mask == 5
    assert len(emitted) == 32 and result["mismatch_trials"] == result["known_vector_failures"] == 0
    assert len(seen) == 32 * 3 * len(chunks)
    assert {ident for _, ident in seen} == {id(chunk) for chunk in chunks}
    assert all(len(record["trials"]) == 16 for record in result["core_records"])
    assert all(row["affinity_verified"] for row in emitted)


@pytest.mark.parametrize("mode", ["mismatch", "exception"])
def test_hash_failure_is_preserved_and_never_retried_or_hides_other_cores(mode):
    chunks, expected = trial_input()
    options = {"wrong" if mode == "mismatch" else "raises": "windows_cng"}
    affinity = Affinity()
    result = d.run_schedule(chunks, expected, factories(chunks, **options), affinity, lambda row: None)
    assert result["all_planned_completed"] and result["completed_trials"] == 32
    assert result["mismatch_trials"] == result["algorithm_disagreement_trials"] == 32
    for core in result["core_records"]:
        assert len(core["trials"]) == 16
        for row in core["trials"]:
            assert row["expected_matches"] == {"openssl": True, "python_sha2": True, "windows_cng": False}
            assert row["digests"]["windows_cng"] == ("0" * 64 if mode == "mismatch" else None)
            assert bool(row["errors"]) is (mode == "exception")


def test_known_vector_failure_is_reported_without_suppressing_fixed_trials():
    chunks, expected = trial_input()
    result = d.run_schedule(chunks, expected, factories(chunks, wrong="openssl", bad_vector=True), Affinity(), lambda row: None)
    assert result["completed_trials"] == 32 and result["known_vector_failures"] == 6


def test_journal_interrupt_finishes_started_core_then_does_not_start_next():
    chunks, expected = trial_input()
    affinity = Affinity()
    seen = []
    def interrupted(row):
        seen.append(row)
        if row["iteration"] == 0:
            raise KeyboardInterrupt("synthetic observer interrupt")
    result = d.run_schedule(chunks, expected, factories(chunks), affinity, interrupted)
    assert len(seen) == result["completed_trials"] == 16
    assert result["completed_cores"] == [0] and affinity.visits == [0] and affinity.mask == 5
    assert result["stop_reason"] == "observer_failure_after_completed_core"
    assert result["unstarted_cores"] == [{"core": 2, "reason": result["stop_reason"]}]
    assert result["observer_errors"][0]["type"] == "KeyboardInterrupt"


def test_affinity_observer_error_preserves_trial_and_finishes_core():
    affinity = Affinity(observe_error=1)
    result = run(affinity)
    assert result["completed_trials"] == 16 and affinity.mask == 5
    assert result["affinity_unverified_trials"] == 1
    assert result["core_records"][0]["trials"][0]["affinity_before_error"]["type"] == "OSError"


def test_soft_budget_is_checked_only_before_new_core():
    times = iter([0, 301])
    affinity = Affinity()
    result = run(affinity, started=0, clock=lambda: next(times))
    assert result["completed_trials"] == 16 and result["completed_cores"] == [0]
    assert result["stop_reason"] == "soft_budget_before_next_core"
    assert affinity.mask == 5 and affinity.restores == [5]


def test_failed_restoration_verification_stops_new_cores_without_retry():
    affinity = Affinity(restore_error=True)
    result = run(affinity)
    assert result["completed_trials"] == 16 and affinity.visits == [0] and affinity.restores == [5]
    assert result["core_errors"] and not result["all_planned_completed"]


def test_digest_context_cleanup_runs_after_hash_exception():
    chunks, expected = trial_input()
    counts = {"closed": 0}
    class Failing:
        def __enter__(self): return self
        def __exit__(self, *args): counts["closed"] += 1
        def update(self, chunk): raise RuntimeError("synthetic update failure")
    algo = factories(chunks)
    algo["windows_cng"] = Failing
    record = d.digest_observation(chunks, algo, expected)
    assert counts["closed"] == 1 and record["digests"]["windows_cng"] is None


def test_block_geometry_uses_fixed_immutable_1mib_chunks():
    payload = b"x" * (d.CHUNK_BYTES + 7)
    chunks = d.blocks(payload)
    assert type(chunks) is tuple and [len(chunk) for chunk in chunks] == [d.CHUNK_BYTES, 7]
    assert all(type(chunk) is bytes for chunk in chunks) and b"".join(chunks) == payload
    with pytest.raises(TypeError): d.blocks(bytearray(payload))


def test_existing_output_is_rejected_before_any_input_or_affinity(monkeypatch, tmp_path):
    monkeypatch.setattr(d, "HERE", tmp_path)
    out = tmp_path / "existing"
    out.mkdir()
    monkeypatch.setattr(d, "load_factories", lambda: pytest.fail("must not start"))
    with pytest.raises(FileExistsError): d.execute(out)


def test_execute_reads_pdf_once_and_persists_complete_synthetic_schedule(monkeypatch, tmp_path):
    payload = b"synthetic PDF"
    chunks = d.blocks(payload)
    algo = factories(chunks)
    affinity = Affinity()
    monkeypatch.setattr(d, "HERE", tmp_path)
    monkeypatch.setattr(d, "EXPECTED_SIZE", len(payload))
    monkeypatch.setattr(d, "EXPECTED_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(d, "load_factories", lambda: (algo, None, b"helper", {}, {}))
    monkeypatch.setattr(d, "CurrentProcessAffinity", lambda: affinity)
    monkeypatch.setattr(d, "runtime_identity", lambda *args: {"files": {"script": {"identity_unchanged": True, "algorithms_agree": True}}})
    reads = []
    def read(path):
        reads.append(path)
        return payload, {"identity_unchanged": True, "file_open_count": 1, "read_calls": 1}
    monkeypatch.setattr(d, "read_once", read)
    output = tmp_path / "run"
    assert d.execute(output) == 0 and reads == [d.PDF]
    finish = json.loads((output / "finish.json").read_text())
    start = json.loads((output / "start.json").read_text())
    rows = [json.loads(line) for line in (output / "trials.jsonl").read_text().splitlines()]
    assert finish["status"] == "completed_without_observed_disagreement" and len(rows) == 32
    assert start["plan"]["trials_per_core"] == 16 and start["plan"]["soft_budget_seconds"] == 300
    assert finish["native_run"] is finish["simulation_run"] is finish["automatic_retry"] is False
