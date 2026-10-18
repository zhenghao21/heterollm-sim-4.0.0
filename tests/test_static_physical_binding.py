"""Only emitter-proven fixed descriptor slots enter static physical replay."""
from copy import deepcopy
from dataclasses import replace

import pytest

from heterollm_sim._physical_binding import StaticPhysicalBinding
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.planner import _gddr_stable_address
from test_dram_core_semantics import config


def pair():
    original = TaskSpec("first.task", "first", "mlp", TaskCategory.COMPUTE,
        demands=(ResourceDemand("memory", 0.0, 64),), metadata={"operator_id": "first.mlp"})
    generation = int.from_bytes(b"\x01first", "big")
    row = {"operation": "write", "physical_owner": "memory", "buffer_id": "first.output",
        "offset_bytes": 32, "byte_count": 64, "allocation_size_bytes": 128,
        "allocation_generation": generation, "alias_generation": generation,
        "address_source": "stable_buffer_tensor_offset", "address": 0}
    attached = replace(original, metadata={**original.metadata,
        "physical_memory_config": config(), "physical_owner": "memory",
        "memory_accesses": (row,), "physical_address_bindings": (dict(row),),
        "physical_allocations": ()})
    return original, attached


def test_known_request_generation_alias_and_rebound_address_are_exact():
    original, attached = pair()
    binding = StaticPhysicalBinding.capture(original, attached,
        identity_is_dynamic=lambda value: value.startswith("first."))
    assert binding is not None
    metadata = deepcopy(attached.metadata)
    binding.bind(metadata, "second", _gddr_stable_address,
        lambda value: value.replace("first.", "second."))
    assert metadata["physical_memory_config"] is attached.metadata["physical_memory_config"]
    expected = int.from_bytes(b"\x01second", "big")
    for collection in ("memory_accesses", "physical_address_bindings"):
        row = metadata[collection][0]
        assert row["allocation_generation"] == row["alias_generation"] == expected
        configured = metadata["physical_memory_config"]
        assert row["address"] == _gddr_stable_address("second.output", 32, 64,
            configured.effective_capacity_bytes, configured.burst_bytes, 128)
    assert attached.metadata["memory_accesses"][0]["allocation_generation"] != expected


@pytest.mark.parametrize("change", ["explicit_generation", "stateful_l2", "direct", "allocation"])
def test_uncovered_declaration_families_use_the_normal_emitter(change):
    original, attached = pair()
    if change == "explicit_generation":
        original = replace(original, metadata={**original.metadata, "input_generation": 7})
    elif change == "stateful_l2":
        attached = replace(attached, metadata={**attached.metadata, "stateful_l2": {}})
    elif change == "direct":
        original = replace(original, metadata={**original.metadata,
            "phase_metadata": {"direct_memory_access": {"component_id": "memory"}}})
    else:
        attached = replace(attached, metadata={**attached.metadata,
            "physical_allocations": ({"buffer_id": "external"},)})
    assert StaticPhysicalBinding.capture(original, attached) is None
