"""Presampling indexes preserve physical ownership, ordering and validation."""
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from heterollm_sim import planner
from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.data_motion import PhysicalRuntimeContext, register_physical_allocations
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.memory_types import make_hbm_config


def _resident_reference(metadata, identity):
    if not identity:
        return False
    if identity in metadata.get("persistent_buffer_ids", ()) or identity.startswith("weights:"):
        return True
    base = identity.removeprefix("tensor:")
    for suffix in (":rank=", ":tp_rank=", ":pp_rank=", ":projection_id="):
        base = base.split(suffix, 1)[0]
    return any(tensor["name"] == base for tensor in metadata.get("f16_weight_tensors", ())) or any(
        metadata.get(key) is not None and str(metadata[key]).strip() == base
        for key in ("weight_buffer_id", "weight_tensor_id", "weight_allocation_id", "tensor_id", "rhs_tensor_id"))


@pytest.mark.parametrize("batch", [0, 1, 256, 1024])
def test_resident_lookup_matches_ownership_and_has_no_cross_call_cache(batch):
    metadata = {"persistent_buffer_ids": [f"kv:r{i}:k" for i in range(batch)],
                "f16_weight_tensors": [{"name": "matrix"}], "rhs_tensor_id": " rhs "}
    before = deepcopy(metadata)
    identities = [None, "", "weights:norm", "temporary_weights", "tensor:matrix:rank=3",
                  "matrix:tp_rank=1", "tensor:rhs:pp_rank=2", "rhs:projection_id=q",
                  "tensor:rhs_cache", *(f"kv:r{i}:k" for i in range(batch))]
    lookup = planner._resident_physical_buffer_lookup(metadata)
    for identity in identities * 2:
        assert lookup(identity) is _resident_reference(metadata, identity)
    assert metadata == before
    metadata["persistent_buffer_ids"].append("new")
    metadata["f16_weight_tensors"][0]["name"] = "other"
    fresh = planner._resident_physical_buffer_lookup(metadata)
    assert fresh("new") and fresh("other") and not fresh("matrix")


def test_resident_lookup_keeps_noncanonical_containment_and_short_circuit():
    class Members:
        def __contains__(self, value):
            return value == "special"
    lookup = planner._resident_physical_buffer_lookup({
        "persistent_buffer_ids": Members(), "f16_weight_tensors": [{"name": "matrix"}, {}]})
    assert lookup("special") and lookup("matrix") and lookup("weights:norm")
    with pytest.raises(KeyError, match="name"):
        lookup("unknown")
    iterator = planner._resident_physical_buffer_lookup({"persistent_buffer_ids": iter(("a", "b"))})
    assert iterator("a") and not iterator("a")


def test_request_index_preserves_row_order_duplicates_owners_and_external_exclusion(monkeypatch):
    kernel = UnifiedEventKernel()
    captured = []
    monkeypatch.setattr(kernel, "register_request_physical_buffers",
                        lambda request, keys: captured.append((request, tuple(keys))))
    task = TaskSpec("index", "request", "index", TaskCategory.MEMORY, metadata={
        "physical_owner": "memory", "physical_external_allocations": (("memory", "a", 1),),
        "physical_allocations": ({"buffer_id": "a", "generation": 0},),
        "persistent_request_buffers": {"first": ["a", "b", "a"], "second": ["b"], "absent": ["missing"]},
        "memory_accesses": (
            {"buffer_id": "b", "allocation_generation": 2},
            {"buffer_id": "a", "allocation_generation": 1},
            {"buffer_id": "b", "allocation_generation": 3, "physical_owner": "other"},
            {"buffer_id": "a", "allocation_generation": 0}),
        "stateful_l2": {"accesses": ({"buffer_id": "a", "allocation_generation": 4},)}})
    before = deepcopy(task)
    kernel._index_physical_allocation_uses((task,))
    assert captured == [
        ("first", (("memory", "a", 0), ("memory", "b", 2), ("other", "b", 3),
                   ("memory", "a", 0), ("memory", "a", 4))),
        ("second", (("memory", "b", 2), ("other", "b", 3)))]
    assert task == before


@pytest.mark.parametrize("buffers", [None, "a", [""], [1]])
def test_request_index_still_rejects_invalid_buffer_declarations(buffers):
    task = TaskSpec("bad", "request", "bad", TaskCategory.MEMORY,
                    metadata={"persistent_request_buffers": {"request": buffers}})
    with pytest.raises(ValueError, match="explicit request and buffer ids"):
        UnifiedEventKernel()._index_physical_allocation_uses((task,))


def test_registration_binds_each_owner_once_and_rechecks_the_next_call(monkeypatch):
    runtime = PhysicalRuntimeContext(capture_details=False)
    original = runtime.runtime
    calls = []
    def counted(config, owner):
        calls.append(owner)
        return original(config, owner)
    monkeypatch.setattr(runtime, "runtime", counted)
    raw = asdict(make_hbm_config(capacity_bytes=4096))
    declarations = tuple({"buffer_id": f"buffer{i}", "size_bytes": 64,
                          "physical_owner": owner} for owner in ("one", "two") for i in range(8))
    task = TaskSpec("register", "request", "register", TaskCategory.MEMORY, metadata={
        "physical_memory_config": raw, "physical_allocations": declarations})
    register_physical_allocations(task, runtime)
    assert calls == ["one", "two"]
    assert all(len(allocator.allocations()) == 8 for allocator in runtime.allocators.values())
    raw["capacity_bytes"] = 8192
    with pytest.raises(ValueError, match="multiple physical geometries"):
        register_physical_allocations(task, runtime)


def test_registration_retains_forward_aliases_and_external_arena_validation():
    task = TaskSpec("aliases", "request", "aliases", TaskCategory.MEMORY, metadata={
        "physical_memory_config": make_hbm_config(capacity_bytes=4096), "physical_owner": "memory",
        "physical_allocations": (
            {"buffer_id": "leaf", "size_bytes": 32, "alias_of": "view"},
            {"buffer_id": "view", "size_bytes": 64, "alias_of": "base"},
            {"buffer_id": "base", "size_bytes": 128})})
    runtime = PhysicalRuntimeContext()
    register_physical_allocations(task, runtime)
    allocator = runtime.allocators["memory"]
    assert allocator.lookup("leaf", 0).base_address == allocator.lookup("base", 0).base_address
    bad = replace(task, metadata={**task.metadata,
        "physical_external_allocations": (("memory", "external", 0),),
        "physical_allocations": ({"buffer_id": "external", "size_bytes": 64, "allocation_arena": "workspace"},)})
    with pytest.raises(ValueError, match="external page-pool allocation"):
        register_physical_allocations(bad, runtime)


def test_kv_template_is_rank_independent_isolated_and_scope_bound(monkeypatch):
    from test_native_external_kv import ingress_scenario
    scenario = ingress_scenario("hbm_tp2", count=1)
    builder = SimpleNamespace(request=scenario.workload.requests[0])
    original = planner._generic_kv_owner_contract
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(planner, "_generic_kv_owner_contract", counted)
    with planner._compilation_scope(scenario):
        plan = planner._parallel_plan(scenario)
        layer = planner._execution_layers(scenario)[0]
        def contract(rank=0, history=6):
            return planner._generic_kv_contract(builder, scenario, plan, plan.ranks[rank], layer,
                token_batch=1, context_tokens=history, history_tokens=history, materialized_tokens=1)
        first, second = contract(), contract(1)
        assert len(calls) == 1
        assert first["owners"][0]["id"].endswith(":rank0:tp0:pp0")
        assert second["owners"][0]["id"].endswith(":rank1:tp1:pp0")
        expected = deepcopy(second)
        first["owners"][0]["read_ranges"].append((100, 1))
        first["owners"][0]["capacity_tokens"] = -1
        assert contract(1) == expected
        with pytest.raises(ValueError, match="declared context capacity"):
            contract(history=10000)
        # Numerically equal but differently typed inputs must still execute
        # the original range/type validation rather than hitting an int key.
        contract()
        with pytest.raises(ValueError, match="query_position must be an integer"):
            contract(history=6.0)
    with planner._compilation_scope(scenario):
        contract()
    assert len(calls) == 4
