"""Only small synthetic reference files; never calls real R33 guard/native/sim."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

HERE = Path(__file__).parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

d = load("guard_observer_test", HERE / "observe_startup_guard.py")
core = load("guard_core_for_test", HERE / "core_digest_stability.py")
s = load("guard_original_evidence_for_test", HERE.parents[1] / "round_023/summarize_ablation.py")


def make(tmp_path, *, wrong=None):
    data = b"synthetic immutable reference"
    path = tmp_path / "evidence.bin"
    path.write_bytes(data)
    updates = []
    def factory(name):
        class Hash:
            def __init__(self): self.value = hashlib.sha256()
            def update(self, block):
                if block == data: updates.append((name, id(block)))
                self.value.update(block)
            def hexdigest(self): return "0" * 64 if name == wrong else self.value.hexdigest()
        return Hash
    observer = d.Observer(s, {name: factory(name) for name in d.NAMES}, core, io.StringIO())
    expected = {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return observer, path, expected, updates


def test_reference_one_open_one_streaming_pass_same_chunk_objects(tmp_path, monkeypatch):
    observer, path, expected, updates = make(tmp_path)
    original = Path.open
    count = {"opens": 0, "reads": []}
    class Stream:
        def __init__(self, inner): self.inner = inner
        def __enter__(self): return self
        def __exit__(self, *args): return self.inner.__exit__(*args)
        def fileno(self): return self.inner.fileno()
        def read(self, n):
            count["reads"].append(n)
            return self.inner.read(n)
    def opened(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        if self == path:
            count["opens"] += 1
            return Stream(value)
        return value
    monkeypatch.setattr(Path, "open", opened)
    assert observer.verify_reference(expected) == expected
    assert count == {"opens": 1, "reads": [1048576, 1048576]}
    assert len(updates) == 3 and len({ident for _, ident in updates}) == 1
    assert observer.report()["full_records"] == []
    ledger = [json.loads(line) for line in observer.stream.getvalue().splitlines()]
    assert len(ledger) == 1 and ledger[0]["h"] == expected["sha256"]


@pytest.mark.parametrize("wrong", ["openssl", "python_sha2", "windows_cng"])
def test_no_algorithm_winner_when_the_same_read_disagrees(tmp_path, wrong):
    observer, path, expected, updates = make(tmp_path, wrong=wrong)
    with pytest.raises(ValueError): observer.verify_reference(expected)
    assert observer.sequence == 1 and len(updates) == 3
    record = observer.report()["full_records"][0]
    assert record["digests"][wrong] == "0" * 64 and not record["algorithms_agree"]
    assert record["comparison"]["normalized_expected"] == expected
    assert record["comparison"]["fields"]["sha256"] is (wrong != "openssl")
    assert record["read_length"] == expected["bytes"] and record["stat_unchanged"]


@pytest.mark.parametrize("field,value", [("path", "wrong-result-path"), ("bytes", 99), ("sha256", "0" * 64)])
def test_reference_dictionary_difference_keeps_each_comparison_field(tmp_path, field, value):
    observer, path, expected, _ = make(tmp_path)
    actual = {**expected, field: value}
    observer.last = {"sequence": 1, "path": str(path), "actual_reference": actual}
    observer.reference = lambda ignored: actual
    with pytest.raises(s.EvidenceError, match="changed evidence"):
        observer.verify_reference(expected)
    comp = observer.report()["full_records"][0]["comparison"]
    assert comp["actual"] == actual and comp["normalized_expected"] == expected
    assert comp["fields"] == {key: key != field for key in ("path", "bytes", "sha256")}


def test_alias_byte_size_uses_original_normalizer_without_mutation(tmp_path):
    observer, path, expected, _ = make(tmp_path)
    alias = {"path": str(path.parent / "." / path.name), "sha256": expected["sha256"], "size_bytes": expected["bytes"]}
    original = dict(alias)
    assert observer.verify_reference(alias) == expected and alias == original


def test_source_reference_declared_size_difference_preserved(tmp_path):
    observer, path, expected, _ = make(tmp_path)
    with pytest.raises(s.EvidenceError, match="source evidence size"):
        observer.verify_source_evidence_reference({**expected, "bytes": expected["bytes"] + 1})
    comp = observer.report()["full_records"][0]["comparison"]
    assert comp["fields"] == {"path": True, "bytes": False, "sha256": True}
    assert comp["actual"]["bytes"] == expected["bytes"]


def test_read_failure_is_not_retried_and_keeps_expected(tmp_path, monkeypatch):
    observer, path, expected, _ = make(tmp_path)
    original = Path.open
    counts = []
    def fail(self, *args, **kwargs):
        if self == path:
            counts.append(self)
            raise OSError("synthetic read failure")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", fail)
    with pytest.raises(OSError): observer.verify_reference(expected)
    assert counts == [path]
    record = observer.report()["full_records"][0]
    assert record["exception"]["message"] == "synthetic read failure" and record["traceback"]
    assert record["comparison"]["normalized_expected"] == expected


def test_process_log_redirect_keeps_function_code_and_other_globals(tmp_path):
    namespace = {"P": Path("original"), "OTHER": object()}
    exec("def function(arg, *, tag='same'):\n    return P, OTHER, arg, tag", namespace)
    original = namespace["function"]
    replacement = d.redirect_process_journal(original, tmp_path)
    assert replacement.__code__ is original.__code__
    assert replacement("arg") == (tmp_path, namespace["OTHER"], "arg", "same")
    assert original("arg")[0] == Path("original")


def test_existing_output_prevents_loading_or_calling_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "HERE", tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(d, "load", lambda *args: pytest.fail("existing output must fail before guard load"))
    with pytest.raises(FileExistsError): d.execute(output)
