"""Only the explicitly bounded simulator UI command can be selected."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

spec = spec_from_file_location("controlled_native_pair", Path(__file__).parents[1] / "tools/run_controlled_native_pair.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("port,expected", [(8764, None), (8765, 8765), (8783, 8783), (8784, 8784), (8785, 8785), (8786, None)])
def test_explicit_ui_port_bounds(port, expected):
    assert module.ui_port(["python.exe", "-m", "heterollm_sim.cli", "ui", "--no-browser", "--port", str(port)]) == expected


@pytest.mark.parametrize("command", [
    ["python.exe", "-m", "unrelated.cli", "ui", "--no-browser", "--port", "8785"],
    ["python.exe", "-m", "heterollm_sim.cli", "ui", "--no-browser", "--port", "8785", "extra"],
])
def test_command_identity_cannot_expand_with_port_range(command):
    assert module.ui_port(command) is None
