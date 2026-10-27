"""Shared address checks name the actual memory and expose capacity failures."""
import hashlib
from pathlib import Path
import sys

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.planner import (
    ScenarioValidationError, _attach_gddr_physical_task, _gddr_stable_address,
)


@pytest.mark.parametrize("kind", ["SRAM", "HBM", "DRAM", None])
def test_allocation_capacity_error_has_real_memory_context(kind):
    context = (dict(physical_kind=kind, component_id="memory0", physical_owner="memory0.fabric")
               if kind is not None else {})
    with pytest.raises(ScenarioValidationError) as caught:
        _gddr_stable_address("logits", 0, 486127488, 320000000, 32, **context)
    error = caught.value
    assert "GDDR" not in str(error)
    assert (kind or "Physical memory") in str(error)
    diagnostic, = error.details["diagnostics"]
    assert diagnostic["code"] == "physical_memory_capacity_exceeded"
    assert diagnostic["stage"] == "capacity_check"
    assert diagnostic["buffer_id"] == "logits"
    assert diagnostic["requested_bytes"] == 486127488
    assert diagnostic["capacity_bytes"] == 320000000
    for name, value in context.items():
        assert diagnostic[name] == value
        assert value in str(error)


@pytest.mark.parametrize("kind", ["SRAM", "HBM", "DRAM", None])
@pytest.mark.parametrize("args", [
    ("", 0, 32, 1024, 32),
    ("buffer", -1, 32, 1024, 32),
    ("buffer", 0, True, 1024, 32),
    ("buffer", 0, 0, 1024, 32),
    ("buffer", 0, 32, 0, 32),
    ("buffer", 0, 32, 1024, 0),
    ("buffer", 16, 32, 1024, 32, 32),
    ("buffer", 0, 32, 1024, 32, -1),
])
def test_invalid_address_geometry_stays_a_correctness_error(kind, args):
    context = (dict(physical_kind=kind, component_id="memory0", physical_owner="memory0.fabric")
               if kind is not None else {})
    with pytest.raises(ValueError) as caught:
        _gddr_stable_address(*args, **context)
    assert type(caught.value) is ValueError
    assert "GDDR" not in str(caught.value)
    assert (kind or "Physical memory") in str(caught.value)
    for value in context.values():
        assert value in str(caught.value)


def test_valid_address_keeps_the_existing_hash_and_extent_math():
    digest = hashlib.blake2b(b"buffer", digest_size=16).digest()
    expected = (int.from_bytes(digest, "big") % ((1024 - 128) // 32 + 1)) * 32 + 16
    assert _gddr_stable_address("buffer", 16, 32, 1024, 32, 128) == expected
    assert _gddr_stable_address("buffer", 16, 32, 1024, 32, 128,
        physical_kind="SRAM", component_id="sram0", physical_owner="sram0.fabric") == expected


def test_sram_backing_capacity_error_propagates_actual_component_and_owner():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from exploring_hbf_native_config import build_native_comparison_scenario

    scenario = build_native_comparison_scenario("llama4_maverick", "HBF+", "SHORT",
        batch_size=4812, tp=1, pp=1, ep=4)
    size = 1203 * 202048 * 2
    task = TaskSpec("head", "request", "lm_head", TaskCategory.COMPUTE,
        demands=(ResourceDemand("sram0.sram_fabric", 0., size),),
        metadata={"target_component": "gpu0", "output_tensor_id": "logits",
            "rank": 0, "tp_rank": 0, "pp_rank": 0,
            "cost_model": {"physical_read_bytes": 0, "physical_write_bytes": size}})
    with pytest.raises(ScenarioValidationError) as caught:
        _attach_gddr_physical_task(task, scenario)
    diagnostic, = caught.value.details["diagnostics"]
    assert diagnostic["physical_kind"] == "SRAM"
    assert diagnostic["component_id"] == "sram0"
    assert diagnostic["physical_owner"] == "sram0.sram_fabric"
    assert diagnostic["buffer_id"] == "tensor:logits:rank=0:tp_rank=0:pp_rank=0"
    assert diagnostic["requested_bytes"] == 486127488
    assert diagnostic["capacity_bytes"] == 320000000
