"""Runtime comparisons keep profiling out of speedup and all simulated fields in parity."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_scan_runtime.py"
    spec = importlib.util.spec_from_file_location("scan_benchmark_comparison_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result():
    return {
        "architecture": "HBF", "model_id": "test-model", "status": "completed",
        "prompt_tokens": 64, "output_tokens": 4,
        "ttft_ns": 100., "tpot_ns": 10.,
        "request_metrics": {"finish_ns": 130.},
        "resource_busy_ns": {"wire": .1}, "storage_traffic": {"pages_read": 4},
        "wall_seconds": 40., "cpu_seconds": 80., "finished_at": "before",
        "python_executable": "python", "gc_thresholds": [7000, 100, 100],
        "profiled": True,
        "cohort_timings": [
            {"completed": 1, "kind": "prefill", "wall_seconds": 10.,
             "cpu_seconds": 20., "profiled": False, "simulated_time_ns": 100.},
            {"completed": 2, "kind": "decode", "wall_seconds": 2.,
             "cpu_seconds": 3., "profiled": False, "simulated_time_ns": 110.},
            {"completed": 3, "kind": "decode", "wall_seconds": 4.,
             "cpu_seconds": 5., "profiled": False, "simulated_time_ns": 120.},
            {"completed": 4, "kind": "decode", "wall_seconds": 20.,
             "cpu_seconds": 40., "profiled": True, "simulated_time_ns": 130.},
        ],
    }


def test_real_windows_are_reported_without_crediting_profiled_total(benchmark):
    baseline = result()
    candidate = deepcopy(baseline)
    candidate.update(wall_seconds=1., cpu_seconds=2., finished_at="after",
                     python_executable="other", gc_thresholds=[9000, 100, 100],
                     source_paths={"hbm": "diagnostic.py"})
    for cohort in candidate["cohort_timings"]:
        cohort["wall_seconds"] /= 2
        cohort["cpu_seconds"] /= 3
    candidate["benchmark_timing"] = benchmark.timing_summary(candidate)
    comparison = benchmark.compare_results(baseline, candidate)
    assert comparison["exactly_equal"]
    assert comparison["speedup"] == {
        "prefill_unprofiled": 2., "decode_unprofiled": 2., "total_unprofiled": None}
    timing = comparison["timing"]["baseline"]
    assert timing["prefill_before_first_token_wall_seconds"] == 10.
    assert timing["wall_to_first_token_seconds"] is None
    assert timing["decode_unprofiled_cohorts"] == [2, 3]
    assert timing["decode_unprofiled_seconds_per_step"] == 3.
    assert timing["total_wall_seconds"] == 40.
    assert timing["total_contains_profiling"]


@pytest.mark.parametrize("changed", ["cohort_time", "finish", "resource", "counter", "unknown", "length"])
def test_every_simulation_value_remains_exact(benchmark, changed):
    baseline = result()
    candidate = deepcopy(baseline)
    if changed == "cohort_time":
        candidate["cohort_timings"][1]["simulated_time_ns"] = math.nextafter(110., math.inf)
    elif changed == "finish":
        candidate["request_metrics"]["finish_ns"] += 1.
    elif changed == "resource":
        candidate["resource_busy_ns"]["wire"] = math.nextafter(.1, math.inf)
    elif changed == "counter":
        candidate["storage_traffic"]["pages_read"] = True
    elif changed == "unknown":
        candidate["storage_traffic"]["wall_seconds"] = 1.
    else:
        candidate["output_tokens"] += 1
    comparison = benchmark.compare_results(baseline, candidate)
    assert not comparison["exactly_equal"]
    assert comparison["difference_count"] == 1
    assert all(value is None for value in comparison["speedup"].values())


def test_different_profile_windows_do_not_compare_decode_rates(benchmark):
    baseline = result()
    candidate = deepcopy(baseline)
    candidate["cohort_timings"][2]["profiled"] = True
    comparison = benchmark.compare_results(baseline, candidate)
    assert comparison["exactly_equal"]
    assert comparison["speedup"]["decode_unprofiled"] is None
    assert comparison["timing"]["candidate"]["decode_unprofiled_cohorts"] == [2]


def test_all_profiled_has_no_unprofiled_decode_rate(benchmark):
    row = result()
    for cohort in row["cohort_timings"]:
        cohort["profiled"] = True
    comparison = benchmark.compare_results(row, deepcopy(row))
    assert comparison["exactly_equal"]
    assert comparison["timing"]["baseline"]["decode_unprofiled_seconds_per_step"] is None
    assert all(value is None for value in comparison["speedup"].values())


def test_total_speedup_requires_two_unprofiled_complete_runs(benchmark):
    baseline = result()
    baseline["profiled"] = False
    for cohort in baseline["cohort_timings"]:
        cohort["profiled"] = False
    candidate = deepcopy(baseline)
    candidate["wall_seconds"] = 20.
    assert benchmark.compare_results(baseline, candidate)["speedup"]["total_unprofiled"] == 2.


def test_cohort_order_is_not_sorted_away(benchmark):
    baseline = result()
    candidate = deepcopy(baseline)
    candidate["cohort_timings"][1:3] = reversed(candidate["cohort_timings"][1:3])
    assert not benchmark.compare_results(baseline, candidate)["exactly_equal"]


def test_startup_wall_is_separate_from_prefill_intervals(benchmark):
    baseline = result()
    candidate = deepcopy(baseline)
    candidate["cohort_timings"][0]["elapsed_wall_seconds"] = 11.
    comparison = benchmark.compare_results(baseline, candidate)
    assert comparison["exactly_equal"]
    timing = comparison["timing"]["candidate"]
    assert timing["prefill_before_first_token_wall_seconds"] == 10.
    assert timing["wall_to_first_token_seconds"] == 11.


def test_identical_incomplete_runs_never_claim_speedup(benchmark):
    row = result()
    row["status"] = "failed"
    comparison = benchmark.compare_results(row, deepcopy(row))
    assert comparison["exactly_equal"]
    assert all(value is None for value in comparison["speedup"].values())


def test_compare_cli_writes_difference_and_does_not_run_simulation(benchmark, tmp_path, monkeypatch, capsys):
    baseline = result()
    candidate = deepcopy(baseline)
    candidate["ttft_ns"] = math.nextafter(candidate["ttft_ns"], math.inf)
    paths = [tmp_path / "before.json", tmp_path / "after.json"]
    for path, row in zip(paths, (baseline, candidate)):
        path.write_text(json.dumps(row), encoding="utf-8")
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--compare", *map(str, paths),
                                      "--result", str(output)])
    def unexpected_run(*args, **kwargs):
        pytest.fail("comparison must not start a simulation")
    monkeypatch.setattr(benchmark, "run_scenario", unexpected_run)
    assert benchmark.main() == 1
    comparison = json.loads(output.read_text(encoding="utf-8"))
    assert not comparison["exactly_equal"]
    assert comparison["differences"][0]["path"] == "$.ttft_ns"
    assert json.loads(capsys.readouterr().out) == comparison


@pytest.mark.parametrize("kind,module_name", [
    ("hbm", "heterollm_sim._hbm_numeric")])
def test_diagnostic_source_loads_before_scenario(benchmark, kind, module_name, tmp_path, monkeypatch):
    source = tmp_path / f"{kind}.py"
    source.write_text("from .memory_types import Operation\nmarker = 'selected source'\n", encoding="utf-8")
    package_name, attribute = module_name.rsplit(".", 1)
    package = sys.modules[package_name]
    previous = sys.modules[module_name]
    # Restore both references after testing a process-global diagnostic load.
    monkeypatch.setitem(sys.modules, module_name, previous)
    monkeypatch.setattr(package, attribute, previous)
    class ScenarioReached(Exception):
        pass
    def scenario(*args):
        selected = sys.modules[module_name]
        assert selected is not previous
        assert selected is getattr(package, attribute)
        assert selected.marker == "selected source"
        assert selected.__file__ == str(source.resolve())
        raise ScenarioReached
    monkeypatch.setattr(benchmark, "build_scan_scenario", scenario)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--architecture", "HBF",
                                      "--result", str(tmp_path / "result.json"),
                                      f"--{kind}-source", str(source)])
    with pytest.raises(ScenarioReached):
        benchmark.main()


def test_no_source_option_does_not_replace_modules(benchmark):
    before = dict(sys.modules)
    assert benchmark.load_sources() == {}
    assert sys.modules == before


def test_invalid_diagnostic_source_restores_original_module(benchmark, tmp_path):
    from numba import config as numba_config
    source = tmp_path / "invalid.py"
    source.write_text("raise RuntimeError('source failed')\n", encoding="utf-8")
    module_name = "heterollm_sim._hbm_numeric"
    previous = sys.modules[module_name]
    previous_cache_dir = numba_config.CACHE_DIR
    with pytest.raises(RuntimeError, match="source failed"):
        benchmark.load_sources(hbm_source=source)
    assert sys.modules[module_name] is previous
    assert sys.modules["heterollm_sim"]._hbm_numeric is previous
    assert numba_config.CACHE_DIR == previous_cache_dir


def test_source_cache_isolated_from_other_import_names_and_reusable(benchmark, tmp_path, monkeypatch):
    from numba import config as numba_config
    source = tmp_path / "numeric.py"
    source.write_text("from numba import njit\n@njit(cache=True)\ndef add(value):\n    return value + 1\n",
                      encoding="utf-8")
    alias = "heterollm_sim._other_diagnostic_name"
    spec = importlib.util.spec_from_file_location(alias, source)
    old = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, alias, old)
    spec.loader.exec_module(old)
    assert old.add(2) == 3  # Create a real cache with the other module environment.
    monkeypatch.delitem(sys.modules, alias)

    canonical = "heterollm_sim._hbm_numeric"
    monkeypatch.setitem(sys.modules, canonical, sys.modules[canonical])
    monkeypatch.setattr(sys.modules["heterollm_sim"], "_hbm_numeric", sys.modules[canonical])
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    previous_cache_dir = numba_config.CACHE_DIR
    assert benchmark.load_sources(hbm_source=source) == {"hbm": str(source.resolve())}
    first = sys.modules[canonical]
    assert numba_config.CACHE_DIR == previous_cache_dir
    assert first.add._cache.cache_path != old.add._cache.cache_path
    assert first.add(2) == 3
    assert sum(first.add._cache_hits.values()) == 0

    benchmark.load_sources(hbm_source=source)
    second = sys.modules[canonical]
    assert second is not first
    assert second.add(2) == 3
    assert sum(second.add._cache_hits.values()) == 1
    assert numba_config.CACHE_DIR == previous_cache_dir


def test_retired_nand_source_option_is_rejected_before_scenario(benchmark, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--architecture", "HBF", "--result", "unused.json",
                                      "--nand-source", "retired.py"])
    monkeypatch.setattr(benchmark, "run_scenario", lambda *args, **kwargs: pytest.fail("must not simulate"))
    with pytest.raises(SystemExit) as caught:
        benchmark.main()
    assert caught.value.code == 2
    assert "unrecognized arguments: --nand-source retired.py" in capsys.readouterr().err


def test_execution_module_preload_uses_selected_packages_available_modules(benchmark, monkeypatch):
    from types import SimpleNamespace
    available = {"heterollm_sim._token_program", "heterollm_sim._io_execution"}
    inspected, imported = [], []
    def find(name):
        inspected.append(name)
        return object() if name in available else None
    def load(name):
        imported.append(name)
        return SimpleNamespace(__file__=name + ".pyd")
    monkeypatch.setattr(benchmark.importlib.util, "find_spec", find)
    monkeypatch.setattr(benchmark.importlib, "import_module", load)
    assert benchmark.preload_execution_modules() == {
        "_token_program": "heterollm_sim._token_program.pyd",
        "_io_execution": "heterollm_sim._io_execution.pyd"}
    assert inspected == ["heterollm_sim." + suffix for suffix in (
        "_token_program", "_physical_task", "_io_execution", "_nand_execution")]
    assert imported == [name for name in inspected if name in available]
