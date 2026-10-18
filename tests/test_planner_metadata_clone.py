"""Captured metadata must survive source-id reuse and retain its alias contract."""
from dataclasses import dataclass

from heterollm_sim.planner import (
    _task_segment_capture_metadata,
    _task_segment_clone_metadata_with_plan,
    _task_segment_omit_overwritten_builtin_branches,
)


@dataclass
class _PhysicalConfig:
    geometry: tuple = ("abc", "def", "ghi")


def _replay(template, plan, *, memo=None, request="next"):
    return _task_segment_clone_metadata_with_plan(
        template, plan, memo=memo, source_request_id="source",
        source_prefix="decode", request_id=request, prefix="decode")


def test_physical_binding_survives_capture_source_id_reused_by_deepcopy_memo():
    binding = {"buffer_id": "source.decode.weights", "byte_count": 16, "offset_bytes": 0}
    source = {"physical_memory_config": _PhysicalConfig(),
              "physical_address_bindings": (binding,)}
    template, plan = _task_segment_capture_metadata(
        source, source_request_id="source", source_prefix="decode")
    # A discarded capture-source dict can be allocated again as the replay
    # memo. Reuse that empty dict explicitly to make the allocator-id collision
    # deterministic, rather than hoping a particular Python heap layout occurs.
    binding.clear()
    replay = _replay(template, plan, memo=binding)
    assert replay["physical_address_bindings"] == (
        {"buffer_id": "next.decode.weights", "byte_count": 16, "offset_bytes": 0},)
    assert isinstance(replay["physical_memory_config"], _PhysicalConfig)
    assert replay["physical_memory_config"] is not template["physical_memory_config"]
    assert template["physical_address_bindings"][0]["buffer_id"] == "source.decode.weights"


def test_metadata_replays_preserve_internal_aliases_and_isolate_mutable_bindings():
    binding = {"buffer_id": "source.decode.weights", "byte_count": 16, "offset_bytes": 0}
    source = {"physical_memory_config": _PhysicalConfig(),
              "physical_address_bindings": (binding, binding),
              "memory_accesses": [binding], "equal_but_distinct_binding": dict(binding)}
    template, plan = _task_segment_capture_metadata(
        source, source_request_id="source", source_prefix="decode")
    first, second = _replay(template, plan), _replay(template, plan, request="later")
    assert first["physical_address_bindings"][0] is first["physical_address_bindings"][1]
    assert first["memory_accesses"][0] is first["physical_address_bindings"][0]
    assert first["equal_but_distinct_binding"] == first["memory_accesses"][0]
    assert first["equal_but_distinct_binding"] is not first["memory_accesses"][0]
    assert first["memory_accesses"][0] is not second["memory_accesses"][0]
    first["memory_accesses"][0]["byte_count"] = 32
    assert first["equal_but_distinct_binding"]["byte_count"] == 16
    assert second["memory_accesses"][0] == {
        "buffer_id": "later.decode.weights", "byte_count": 16, "offset_bytes": 0}
    assert template["memory_accesses"][0]["byte_count"] == 16
    assert binding["byte_count"] == 16


def test_overwrite_pruning_keeps_descriptor_aliases_routes_and_key_order():
    descriptor = {"buffer_id": "source.decode.value", "byte_count": 16}
    source = {"cost_model": {"cache": {"old_bytes": 16}},
        "memory_accesses": [descriptor], "physical_address_bindings": (descriptor,),
        "hbf_host_staging_by_owner": {"memory:hbf0": {"routes": {
            "read": ({"resource_capacities": {"host0.link": 2}},)}}}}
    template, plan = _task_segment_capture_metadata(
        source, source_request_id="source", source_prefix="decode")
    pruned = _task_segment_omit_overwritten_builtin_branches(plan, {"cost_model"})
    first = _replay(template, pruned)
    second = _replay(template, pruned, request="later")
    # The slot replaces the borrowed old value before a TaskSpec is published.
    assert first["cost_model"] is template["cost_model"]
    first["cost_model"] = {"cache": {"old_bytes": 32}}
    assert tuple(first) == tuple(source)
    assert first["memory_accesses"][0] is first["physical_address_bindings"][0]
    assert first["memory_accesses"][0] is not second["memory_accesses"][0]
    assert first["memory_accesses"][0] is not template["memory_accesses"][0]
    capacities = first["hbf_host_staging_by_owner"]["memory:hbf0"]["routes"]["read"][0]["resource_capacities"]
    capacities["host0.link"] = 9
    assert second["hbf_host_staging_by_owner"]["memory:hbf0"]["routes"]["read"][0]["resource_capacities"] == {"host0.link": 2}
    assert template["cost_model"] == {"cache": {"old_bytes": 16}}


def test_overwrite_pruning_keeps_unknown_copy_hooks():
    template, plan = _task_segment_capture_metadata(
        {"cost_model": {"custom": _PhysicalConfig()}},
        source_request_id="source", source_prefix="decode")
    pruned = _task_segment_omit_overwritten_builtin_branches(plan, {"cost_model"})
    assert pruned == plan
    actual = _replay(template, pruned)
    assert actual["cost_model"] is not template["cost_model"]
    assert actual["cost_model"]["custom"] is not template["cost_model"]["custom"]
