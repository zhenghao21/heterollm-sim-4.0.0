"""Runtime routing checks use only file metadata; no simulator is imported."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_fast_mode.py"


def binding_function():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "bound_runtime_command")
    namespace = {"argparse": argparse, "json": json, "Path": Path,
                 "sys": SimpleNamespace(executable="test-python"),
                 "__file__": str(SCRIPT)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SCRIPT), "exec"),
         namespace)
    return namespace[function.name]


def create_binding(tmp_path, *, complete=True, schema="fast-validation-runtime/v1"):
    retained = tmp_path / "retained runtime"
    (retained / "scripts").mkdir(parents=True)
    script = retained / "scripts" / "benchmark_fast_mode.py"
    script.write_text("# retained diagnostic\n", encoding="utf-8")
    if complete:
        package = retained / "src" / "heterollm_sim"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    (results / "runtime_binding.json").write_text(json.dumps({
        "schema": schema, "runtime_root": str(retained)}), encoding="utf-8")
    return results / "one.json", script


def test_binding_preserves_complete_request_and_absolute_output_arguments(tmp_path):
    result, retained_script = create_binding(tmp_path)
    argv = ["--mode", "detailed", "--architecture", "HBF", "--prompt", "65536",
            "--output", "5120", "--result", str(result),
            "--progress-file", str(result.parent / "progress.json"),
            "--stop-file", str(result.parent / "stop.request"), "--report-timing"]
    assert binding_function()(argv) == [
        "test-python", "-u", str(retained_script), *argv]


def test_unbound_diagnostics_and_read_only_comparison_stay_local(tmp_path):
    route = binding_function()
    assert route(["--compare", "a.json", "b.json", "c.json", "d.json"]) is None
    assert route(["--result", str(tmp_path / "ordinary.json")]) is None


@pytest.mark.parametrize("complete,schema", [
    (False, "fast-validation-runtime/v1"), (True, "unknown-schema")])
def test_broken_binding_fails_instead_of_silently_using_current_source(
        tmp_path, complete, schema):
    result, _ = create_binding(tmp_path, complete=complete, schema=schema)
    with pytest.raises(ValueError):
        binding_function()(["--mode", "detailed", "--result", str(result)])


def test_self_binding_does_not_launch_another_process(tmp_path):
    result = tmp_path / "one.json"
    (tmp_path / "runtime_binding.json").write_text(json.dumps({
        "schema": "fast-validation-runtime/v1", "runtime_root": str(SCRIPT.parents[1])}),
        encoding="utf-8")
    assert binding_function()(["--mode", "detailed", "--result", str(result)]) is None
