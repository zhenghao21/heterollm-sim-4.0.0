from pathlib import Path
import pytest
from tools import check_freeze_manifest as freeze


def test_create_does_not_overwrite_existing_manifest(tmp_path):
    target = tmp_path / "freeze.json"
    target.write_text("original")
    with pytest.raises(SystemExit, match="overwrite"):
        freeze.create(target)
    assert target.read_text() == "original"


def test_create_missing_proof_never_writes_frozen_file(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "_artifact_inputs", lambda: {})
    monkeypatch.setattr(freeze, "probe_hardware", lambda: {})
    monkeypatch.setattr(freeze, "_hardware_errors", lambda _: [])
    monkeypatch.setattr(freeze, "_scenario_manifest", lambda: {})
    target = tmp_path / "freeze.json"
    with pytest.raises(SystemExit, match="semantic proof missing"):
        freeze.create(target)
    assert not target.exists()


def test_source_inventory_keeps_empty_inputs_to_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    directory = tmp_path / "src"
    directory.mkdir()
    source = directory / "empty.py"
    source.touch()
    assert source in freeze._source_files()
    with pytest.raises(ValueError, match="empty freeze input"):
        freeze.digest(source)


def test_invalid_cells_do_not_increase_metric_coverage():
    from tools.generalization_acceptance_matrix import _metric_denominators
    cells = [{"status": "invalid", "metrics": {"ttft_ms": {"status": "measured"}}}]
    assert _metric_denominators(cells, 2)["ttft_ms"]["coverage"] == 0


def test_observer_summary_uses_process_blocks():
    from tools.check_token_observer import summarize
    rows = []
    for block, enabled in enumerate((0, 1, 1, 0)):
        for repeat in range(3):
            rows.append({"block": block, "enabled": enabled, "timings": {
                "prompt_ms": 2.0, "predicted_ms": 7.0, "predicted_n": 8}})
    result = summarize(rows)
    assert result["ttft_ms"]["median_change_pct"] == 0
    assert len(result["ttft_ms"]["off_block_medians_ms"]) == 2
