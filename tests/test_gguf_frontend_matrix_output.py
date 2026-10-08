"""Validation directories must be explicit and never alter an earlier report."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location("recover_gguf_matrix", ROOT / "tools/recover_gguf_frontend_matrix.py")
recovery = module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def prepared_directory(directory):
    directory.mkdir()
    (directory / "base").mkdir()
    write(directory / "cases.json", {"cases": [{"case_id": "example", "preset_id": "example"}]})
    scenario = {"model": {"metadata": {}, "graph": {"attributes": {}, "tensors": []}}}
    write(directory / "base/scenario_example_512_128.json", scenario)
    for mode in ("off", "on"):
        write(directory / f"scenario_example_graph_{mode}.json", scenario)


def run_node(*arguments):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    return subprocess.run([node, str(ROOT / "tools/run_gguf_frontend_matrix.cjs"), *arguments],
                          capture_output=True, text=True, encoding="utf-8", timeout=30)


def test_frontend_check_only_reads_selected_output_without_writing(tmp_path):
    directory = tmp_path / "new comparison"
    prepared_directory(directory)
    before = {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    result = run_node("--output-dir", str(directory), "--phase", "all", "--check-only")
    assert result.returncode == 0, result.stderr
    checked = json.loads(result.stdout)
    assert checked["pending"] == [{"case_id": "example", "mode": "off"},
                                  {"case_id": "example", "mode": "on"}]
    assert {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("arguments", [
    ("--output-dir",), ("--output-dir", ""), ("--output-dir", "--phase", "off"),
])
def test_frontend_output_argument_requires_value(arguments):
    result = run_node(*arguments)
    assert result.returncode != 0
    assert "--output-dir requires a value" in result.stderr


def test_frontend_output_directory_must_exist(tmp_path):
    result = run_node("--output-dir", str(tmp_path / "missing"), "--phase", "off", "--check-only")
    assert result.returncode != 0
    assert "existing prepared validation directory" in result.stderr
    assert not (tmp_path / "missing").exists()


def test_submit_only_requires_one_service_per_job_before_any_submission(tmp_path):
    directory = tmp_path / "dispatch comparison"
    prepared_directory(directory)
    args = ("--output-dir", str(directory), "--phase", "all", "--submit-only", "--check-only")
    rejected = run_node(*args, "--ports", "8794")
    assert rejected.returncode != 0
    assert "a separate free service port" in rejected.stderr
    accepted = run_node(*args, "--ports", "8794,8795")
    assert accepted.returncode == 0, accepted.stderr
    assert not (directory / "ui_runs.json").exists()
    rejected = run_node(*args, "--resume", "--ports", "8794,8795")
    assert rejected.returncode != 0
    assert "cannot recover existing jobs" in rejected.stderr


@pytest.mark.parametrize("flag", ["--output-dir", "--output", None])
def test_recovery_uses_selected_directory_preserving_terminal_bytes(tmp_path, monkeypatch, flag):
    output = tmp_path / "new comparison"
    output.mkdir()
    result_path = output / "ui_example_graph_on_result.json"
    original = b'{"job_id":"job","status":"completed","report":{"kept": true}}\n'
    result_path.write_bytes(original)
    created = output / "created.json"
    submitted = output / "submitted.json"
    write(created, {"job_id": "job"})
    write(submitted, {"scenario": {}})
    write(output / "ui_runs.json", {"runs": [{"case_id": "example", "mode": "on", "job_id": "job",
          "status": "completed", "url": "http://127.0.0.1:8794/", "job_created_path": str(created),
          "submission_path": str(submitted), "result_path": str(result_path)}]})
    monkeypatch.setattr(recovery, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(recovery, "observe", lambda *args: pytest.fail("Completed job must not be fetched"))
    args = [flag, str(output)] if flag else []
    assert recovery.main([*args, "--once"]) == 0
    assert result_path.read_bytes() == original
    assert json.loads((output / "ui_runs.json").read_text())["runs"][0]["status"] == "completed"


@pytest.mark.parametrize("arguments", [
    ["--output-dir", ""], ["--timeout", "nan"], ["--interval", "inf"], ["--interval", "0"],
])
def test_recovery_invalid_arguments_fail_before_io(arguments):
    with pytest.raises(SystemExit) as error:
        recovery.main(arguments)
    assert error.value.code == 2
