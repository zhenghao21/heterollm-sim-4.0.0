"""Diagnostic observations remain exact and never become speed measurements."""
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def capture():
    sys.path.insert(0, str(ROOT / "scripts"))
    from capture_physical_transactions import PhysicalTransactionCapture, compare_transaction_files, exact_value
    return PhysicalTransactionCapture, compare_transaction_files, exact_value


def test_observations_keep_float_bits_and_native_array_bits(capture):
    _, _, exact = capture
    assert exact(-0.) != exact(0.)
    assert exact(1.) != exact(np.nextafter(1., np.inf).item())
    assert exact(np.array([-0., 1.])) != exact(np.array([0., 1.]))
    assert exact(np.array([1], dtype=np.int64)) != exact(np.array([1], dtype=np.float64))
    assert exact(np.array([[1, 2]])) != exact(np.array([1, 2]))
    assert exact(1) != exact(True)
    assert exact(1) != exact(1.)


def test_observations_keep_tuple_keys_and_dataclass_fields(capture):
    _, _, exact = capture
    @dataclass
    class Transaction:
        completion: float
        _state: tuple
    assert exact(Transaction(1., (-0.,))) == {
        "type": "test_observations_keep_tuple_keys_and_dataclass_fields.<locals>.Transaction",
        "fields": {"completion": {"float64": "0x1.0000000000000p+0"},
                   "_state": {"tuple": [{"float64": "-0x0.0p+0"}]}}}
    assert exact({(1, 2): 3, "a": 4}) == exact({"a": 4, (1, 2): 3})
    assert exact((1, 2)) != exact([1, 2])


@pytest.mark.parametrize("difference", ["value", "order", "missing"])
def test_complete_transaction_trace_is_compared_in_order(capture, tmp_path, difference):
    _, compare, _ = capture
    before = [{"kind": "read", "value": 1}, {"kind": "write", "value": 2}]
    after = deepcopy(before)
    if difference == "value":
        after[1]["value"] = 3
    elif difference == "order":
        after.reverse()
    else:
        after.pop()
    paths = [tmp_path / "before.jsonl", tmp_path / "after.jsonl"]
    for path, rows in zip(paths, (before, after)):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert not compare(*paths)["exactly_equal"]
    assert compare(paths[0], paths[0]) == {"exactly_equal": True, "record_count": 2}


def test_implementation_root_is_selected_before_any_backend_import(tmp_path):
    package = tmp_path / "heterollm_sim"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('selected diagnostic package first')\n",
                                        encoding="utf-8")
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "benchmark_scan_runtime.py"),
                             "--implementation-root", str(tmp_path), "--help"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "selected diagnostic package first" in result.stderr


def test_physical_capture_restores_entry_points_and_keeps_every_page(capture, tmp_path):
    from heterollm_sim import data_motion
    from heterollm_sim.nand_core import NandCore
    from heterollm_sim.dram_core import DramCore
    Capture, _, _ = capture
    original = (data_motion.resolve_physical_task, NandCore.submit, DramCore.submit_many)
    target = tmp_path / "observations.jsonl"
    with Capture(target) as observed:
        observed.write("pages", (np.array([0, 4096]), np.array([1., 2.]), -0.))
        assert NandCore.submit is not original[1]
        assert DramCore.submit_many is not original[2]
    assert (data_motion.resolve_physical_task, NandCore.submit, DramCore.submit_many) == original
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["value"]["tuple"][1]["shape"] == [2]
    assert rows[0]["value"]["tuple"][2] == {"float64": "-0x0.0p+0"}


def test_captured_execution_can_never_establish_speedup(capture):
    import benchmark_scan_runtime as benchmark
    before = {"status": "completed", "wall_seconds": 10., "cpu_seconds": 10., "profiled": False,
              "cohort_timings": [{"completed": 1, "kind": "prefill", "wall_seconds": 5.,
                                  "cpu_seconds": 5., "profiled": False, "simulated_time_ns": 1.},
                                 {"completed": 2, "kind": "decode", "wall_seconds": 5.,
                                  "cpu_seconds": 5., "profiled": False, "simulated_time_ns": 2.}]}
    after = deepcopy(before)
    after["transaction_capture"] = "trace.jsonl"
    result = benchmark.compare_results(before, after)
    assert result["exactly_equal"]
    assert all(item is None for item in result["speedup"].values())
    assert any("Transaction capture" in item for item in result["notes"])


@pytest.mark.parametrize("accepted", [False, True])
def test_task_local_host_controller_keeps_same_transaction_observation(capture, tmp_path, monkeypatch, accepted):
    import gzip
    from types import SimpleNamespace
    from heterollm_sim import _io_execution
    from heterollm_sim.memory_types import AccessRequest, Operation, TransactionResult
    Capture, compare, _ = capture
    request = AccessRequest("host", Operation.READ, 0, 256, 1.)
    result = TransactionResult("host", Operation.READ, 1., 2., 256, 256)
    core = SimpleNamespace(config=SimpleNamespace(metadata={}))
    scope = SimpleNamespace(core=core)
    monkeypatch.setattr(_io_execution, "execute_compiled", lambda *args, **kwargs: result if accepted else None)
    path = tmp_path / "host.jsonl.gz"
    with Capture(path):
        assert _io_execution.execute_compiled(core, request, _scope=scope) is (result if accepted else None)
    with gzip.open(path, "rt", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    assert len(rows) == int(accepted)
    if accepted:
        assert rows[0]["kind"] == "dram_request"
    assert compare(path, path)["exactly_equal"]
