"""Report readers see complete artifacts while Windows sharing conflicts retry."""
import builtins
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest


@pytest.fixture
def report():
    source = Path(__file__).resolve().parents[1] / "scripts" / "b200_hbm_hbf_scan_report.py"
    spec = importlib.util.spec_from_file_location("b200_atomic_report_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def windows_io(report, monkeypatch, replace):
    monkeypatch.setattr(report, "os", SimpleNamespace(name="nt", replace=replace))
    monkeypatch.setattr(report.time, "sleep", lambda _seconds: None)


def test_atomic_text_keeps_old_file_complete_until_replace_succeeds(report, tmp_path, monkeypatch):
    target = tmp_path / "report.md"
    target.write_text("old complete report", encoding="utf-8")
    replace = report.os.replace
    attempts = []

    def temporarily_busy(source, destination):
        assert target.read_text(encoding="utf-8") == "old complete report"
        assert Path(source).read_text(encoding="utf-8") == "new complete report"
        attempts.append(source)
        if len(attempts) < 3:
            raise OSError(22, "temporarily mapped")
        return replace(source, destination)

    windows_io(report, monkeypatch, temporarily_busy)
    report._atomic_text(target, "new complete report")
    assert len(attempts) == 3
    assert target.read_text(encoding="utf-8") == "new complete report"
    assert not list(tmp_path.glob(".report-*"))


@pytest.mark.parametrize("errno,winerror", [(13, None), (22, None), (5, 32), (5, 33), (5, 1224)])
def test_persistent_windows_conflict_is_bounded_and_cleans_temporary(report, tmp_path, monkeypatch, errno, winerror):
    target = tmp_path / "report.md"
    target.write_text("old intact", encoding="utf-8")
    error = OSError(errno, "still busy")
    if winerror is not None:
        error.winerror = winerror
    attempts = []

    def busy(_source, _destination):
        attempts.append(1)
        assert target.read_text(encoding="utf-8") == "old intact"
        raise error

    windows_io(report, monkeypatch, busy)
    with pytest.raises(OSError) as raised:
        report._atomic_text(target, "never published")
    assert raised.value is error and len(attempts) == 5
    assert target.read_text(encoding="utf-8") == "old intact"
    assert not list(tmp_path.glob(".report-*"))


@pytest.mark.parametrize("platform,errno", [("nt", 28), ("nt", 5), ("posix", 13), ("posix", 22)])
def test_unrelated_io_errors_are_not_retried(report, tmp_path, monkeypatch, platform, errno):
    target = tmp_path / "report.md"
    target.write_text("old", encoding="utf-8")
    calls = []
    error = OSError(errno, "real failure")

    def fail(_source, _destination):
        calls.append(1)
        raise error

    monkeypatch.setattr(report, "os", SimpleNamespace(name=platform, replace=fail))
    monkeypatch.setattr(report.time, "sleep", lambda _seconds: pytest.fail("Must not retry this error"))
    with pytest.raises(OSError) as raised:
        report._atomic_text(target, "new")
    assert raised.value is error and len(calls) == 1
    assert target.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".report-*"))


def test_failed_writer_never_truncates_visible_artifact(report, tmp_path, monkeypatch):
    target = tmp_path / "comparison.png"
    target.write_bytes(b"old complete image")
    windows_io(report, monkeypatch, report.os.replace)
    calls = []
    error = OSError(28, "disk full")

    def writer(temporary):
        calls.append(temporary)
        temporary.write_bytes(b"partial")
        raise error

    with pytest.raises(OSError) as raised:
        report._atomic_file(target, writer)
    assert raised.value is error and len(calls) == 1
    assert target.read_bytes() == b"old complete image"
    assert not list(tmp_path.glob(".report-*"))


def test_transient_sibling_write_retries_without_publishing_partial_content(report, tmp_path, monkeypatch):
    target = tmp_path / "comparison.png"
    target.write_bytes(b"old complete image")
    windows_io(report, monkeypatch, report.os.replace)
    calls = []

    def writer(temporary):
        assert target.read_bytes() == b"old complete image"
        calls.append(temporary)
        temporary.write_bytes(b"partial" if len(calls) < 3 else b"new complete image")
        if len(calls) < 3:
            raise OSError(22, "temporary sibling mapping")

    report._atomic_file(target, writer)
    assert len(calls) == 3
    assert target.read_bytes() == b"new complete image"
    assert not list(tmp_path.glob(".report-*"))


def fixture_scan(tmp_path):
    (tmp_path / "points").mkdir()
    manifest = {
        "models": [{"model_id": "small", "label": "Small", "parameters_b": .5}],
        "architectures": ["HBM", "HBF"], "prompts": [65536], "outputs": [1024],
    }
    row = {"point_id": "small-hbm", "model_id": "small", "architecture": "HBM",
           "prompt_tokens": 65536, "output_tokens": 1024,
           "status": "forecast_completed", "ttft_ns": 2e9, "tpot_ns": 3e6}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "points" / "small-hbm.json").write_text(json.dumps(row), encoding="utf-8")
    return manifest, row


def test_real_figures_publish_from_siblings_when_targets_cannot_be_opened_for_write(report, tmp_path, monkeypatch):
    from PIL import Image
    from matplotlib.figure import Figure

    manifest, row = fixture_scan(tmp_path)
    targets = {tmp_path / "comparison.png", tmp_path / "comparison.svg"}
    for target in targets:
        target.write_bytes(b"old image remains complete")
    real_open = builtins.open
    real_savefig = Figure.savefig
    saved = []

    def protected_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file) in targets and any(value in mode for value in "wax+"):
            pytest.fail("Visible figures must never be opened for direct writing")
        return real_open(file, mode, *args, **kwargs)

    def savefig(figure, filename, *args, **kwargs):
        path = Path(filename)
        assert path.parent == tmp_path and path not in targets
        assert path.suffix in {".png", ".svg"}
        saved.append(path)
        return real_savefig(figure, filename, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", protected_open)
    monkeypatch.setattr(Figure, "savefig", savefig)
    report._plot(tmp_path, manifest, [row], 2)
    assert len(saved) == 2
    with Image.open(tmp_path / "comparison.png") as image:
        image.verify()
    assert ET.parse(tmp_path / "comparison.svg").getroot().tag.endswith("svg")
    assert not list(tmp_path.glob(".report-*"))


def test_csv_and_result_json_are_complete_before_atomic_replacement(report, tmp_path, monkeypatch):
    fixture_scan(tmp_path)
    for name in ("summary.csv", "results.json"):
        (tmp_path / name).write_text("old complete", encoding="utf-8")
    replace = report.os.replace
    published = []

    def verify(source, destination):
        destination = Path(destination)
        if destination.name in {"summary.csv", "results.json"}:
            assert destination.read_text(encoding="utf-8") == "old complete"
            if destination.suffix == ".json":
                assert json.loads(Path(source).read_text(encoding="utf-8"))["counts"]["completed"] == 1
            else:
                assert Path(source).read_bytes().startswith(b"\xef\xbb\xbf")
                with Path(source).open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                assert len(rows) == 1 and rows[0]["point_id"] == "small-hbm"
            published.append(destination.name)
        return replace(source, destination)

    monkeypatch.setattr(report.os, "replace", verify)
    monkeypatch.setattr(report, "_plot", lambda *_args: None)
    report.write_report(tmp_path)
    assert set(published) == {"summary.csv", "results.json"}
    assert not list(tmp_path.glob(".report-*"))


def test_plot_closes_figure_when_saving_really_fails(report, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    manifest, row = fixture_scan(tmp_path)
    initial_figures = plt.get_fignums()
    target = tmp_path / "comparison.png"
    target.write_bytes(b"old complete image")

    def fail(_figure, _filename, *_args, **_kwargs):
        raise OSError(28, "disk full")

    monkeypatch.setattr(Figure, "savefig", fail)
    with pytest.raises(OSError, match="disk full"):
        report._plot(tmp_path, manifest, [row], 2)
    assert plt.get_fignums() == initial_figures
    assert target.read_bytes() == b"old complete image"
    assert not list(tmp_path.glob(".report-*"))
