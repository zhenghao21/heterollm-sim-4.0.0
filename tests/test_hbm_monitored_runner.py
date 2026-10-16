"""A new monitored run must not overwrite incomplete earlier diagnostics."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_hbm_monitored import require_new_round


@pytest.mark.parametrize("marker", ["plan.json", "source_version.json", "progress", "serving/point.json"])
def test_incomplete_or_completed_round_cannot_be_reused(tmp_path, marker):
    target = tmp_path / marker
    if marker == "progress":
        target.mkdir()
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("unchanged", encoding="utf-8")
    with pytest.raises(ValueError, match="new round directory"):
        require_new_round(tmp_path)
    if target.is_file():
        assert target.read_text(encoding="utf-8") == "unchanged"


def test_new_directory_is_allowed_without_creating_it(tmp_path):
    target = tmp_path / "round2"
    require_new_round(target)
    assert not target.exists()
