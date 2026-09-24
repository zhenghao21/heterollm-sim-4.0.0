from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from tools import multiworkload_architecture_matrix as matrix
from tools import phase1_matrix_runner as phase1


def _manifest(tmp_path: Path) -> Path:
    output = tmp_path / "run"
    output.mkdir()
    payload = {
        "source": matrix.source_identity(),
        "workloads": [["L1_short_low", 128, 32, 1], ["L6_long_decode", 2048, 512, 16]],
        "groups": [{"id": "I1", "cases": ["kv_hbm", "kv_hbf"], "values": ["HBM", "HBF"]}],
    }
    (output / "manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return output


def test_phase1_weights_keep_long_decode_single_slot():
    assert phase1._weight("L1_short_low") == 1
    assert phase1._weight("L4_long_context") == 3
    assert phase1._weight("L6_long_decode") == 8
    assert phase1._weight("L8_short_high_batch") == 6


def test_phase1_job_inventory_is_manifest_driven(tmp_path):
    output = _manifest(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    jobs = phase1._jobs(manifest, ["L6"])
    assert [job.cell_id for job in jobs] == [
        "L6_long_decode__kv_hbm",
        "L6_long_decode__kv_hbf",
    ]
    assert all(job.weight == 8 for job in jobs)
    manifest["cancelled_cells"] = ["L6_long_decode__kv_hbm"]
    assert [job.cell_id for job in phase1._jobs(manifest, ["L6"])] == ["L6_long_decode__kv_hbf"]


def test_worker_rebinds_qwen_execution_to_bounded_exact_cache(tmp_path, monkeypatch):
    output = _manifest(tmp_path)
    captured = {}

    class FakeProvider:
        def __init__(self, scenario, *, template_cache_entries, leaf_cache_entries):
            captured["cache"] = (template_cache_entries, leaf_cache_entries)
            self._compilation_context = SimpleNamespace(
                metadata_source_cache_entries=65536,
                artifact_workload_cache_entries=65536,
            )

        def _rebind_control_plane_successor(self, source, scenario):
            self._compilation_context = SimpleNamespace(
                metadata_source_cache_entries=65536,
                artifact_workload_cache_entries=65536,
            )

    def fake_run_scenario(scenario, *, retention_policy, batch_lowerer):
        captured["lowerer"] = batch_lowerer
        captured["retention_policy"] = retention_policy
        old_context = batch_lowerer._compilation_context
        batch_lowerer._rebind_control_plane_successor(scenario, "mapped")
        assert batch_lowerer._compilation_context is not old_context
        assert batch_lowerer._compilation_context.metadata_source_cache_entries == 256
        assert batch_lowerer._compilation_context.artifact_workload_cache_entries == 256
        return object()

    monkeypatch.setattr(phase1, "TopologyAwareBatchCostProvider", FakeProvider)
    monkeypatch.setattr(phase1.matrix.q, "run_scenario", fake_run_scenario)

    def fake_run_cell(variant, workload, run_dir):
        phase1.matrix.q.run_scenario("scenario", retention_policy="aggregate")

    monkeypatch.setattr(phase1.matrix, "run_cell", fake_run_cell)
    args = phase1.build_parser().parse_args([
        "--output", str(output),
        "--worker-case", "kv_hbm",
        "--worker-load-id", "L1_short_low",
        "--template-cache-entries", "8",
        "--leaf-cache-entries", "256",
    ])

    assert phase1._bounded_worker(args) == 0
    assert captured["cache"] == (8, 256)
    assert captured["retention_policy"] == "aggregate"
    assert isinstance(captured["lowerer"], FakeProvider)


def test_cache_limits_survive_real_provider_rebind(tmp_path, monkeypatch):
    from heterollm_sim.reference import build_reference_scenario

    scenario = build_reference_scenario()
    mapped = replace(scenario, name=scenario.name + "-mapped")
    output = _manifest(tmp_path)

    def fake_run(scenario, *, retention_policy, batch_lowerer):
        before = batch_lowerer._compilation_context
        batch_lowerer._rebind_control_plane_successor(scenario, mapped)
        context = batch_lowerer._compilation_context
        assert context is not before and context.scenario is mapped
        assert context.leaf_cache_entries == 256
        assert batch_lowerer._templates.max_entries == 8
        assert context.metadata_source_cache_entries == 256
        assert context.artifact_workload_cache_entries == 256
        return object()

    monkeypatch.setattr(phase1.matrix.q, "run_scenario", fake_run)
    monkeypatch.setattr(phase1.matrix, "run_cell", lambda *args:
        phase1.matrix.q.run_scenario(scenario, retention_policy="aggregate"))
    args = phase1.build_parser().parse_args([
        "--output", str(output), "--worker-case", "kv_hbm",
        "--worker-load-id", "L1_short_low",
    ])
    assert phase1._bounded_worker(args) == 0
