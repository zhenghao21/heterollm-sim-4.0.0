"""Logical tied tensors share storage only with a proven identical local view."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from b200_hbm_hbf_scan_config import build_scan_scenario
from heterollm_sim.control_plane_planner import _derive_requirements, plan_runtime_placement
from heterollm_sim.ir import LayerSpec, ModelSpec, build_model_graph_from_layer_specs
from heterollm_sim.parallel import build_parallel_plan
from heterollm_sim.planner import (
    _shared_input_output_weight_storage, compile_scenario,
)


def _planned(strategy="HBF", model_id="qwen2_5-0_5b", *, small=False):
    scenario = build_scan_scenario(model_id, strategy, 1, 1)
    if small:
        layer = LayerSpec("dense0", "dense", 32, 64, 4, kv_heads=2, dtype="bf16")
        model = ModelSpec(name="small-tied-storage", graph=build_model_graph_from_layer_specs(
            "small-tied-storage", (layer,), vocabulary_size=64,
            max_sequence_length=32, tie_word_embeddings=True))
        scenario = replace(scenario, model=model, placement=replace(scenario.placement,
            model_name=model.name, metadata={}, tensor_bytes={}, tensor_to_component={},
            parallel=replace(scenario.placement.parallel, layer_to_stage={"dense0": 0})))
        requirements = _derive_requirements(scenario)
        memory = "hbf0" if strategy == "HBF" else "hbm0"
        policy = {"operator_targets": {row.mapping_key: "gpu0" for row in requirements
                                      if row.mapping_key and not row.state_tensor},
                  "weight_tensor_targets": {row.tensor_id: memory for row in requirements
                                            if row.tensor_id and row.tensor_bytes > 0 and not row.state_tensor},
                  "kv_cache_target": memory, "design_prefill_tokens": 1,
                  "design_decode_batch_size": 1}
        scenario = replace(scenario, placement=replace(scenario.placement,
            metadata={"control_plane": {"policy": {"options": policy}}}))
    decision = plan_runtime_placement(scenario)
    assert decision.fully_placed
    return decision.apply(scenario)


@pytest.mark.parametrize("strategy", ["HBM", "HBF"])
@pytest.mark.parametrize("model_id", ["qwen2_5-0_5b", "qwen2_5-1_5b", "qwen2_5-3b", "qwen2_5-7b"])
def test_formal_models_keep_tied_and_untied_storage_distinct(strategy, model_id):
    scenario = _planned(strategy, model_id)
    plan = build_parallel_plan(scenario)
    rank = plan.ranks[0]
    for logical in ("embedding_weights", "lm_head_weights"):
        actual = _shared_input_output_weight_storage(scenario, plan, rank, logical, "gpu0")
        if model_id == "qwen2_5-7b":
            assert actual is None
        else:
            assert actual == ("embedding_weights", scenario.model.embedding_weight_bytes)


@pytest.mark.parametrize("strategy", ["HBM", "HBF"])
def test_physical_lowering_keeps_one_complete_table_and_both_logical_tensor_ids(strategy):
    scenario = _planned(strategy, small=True)
    tasks = compile_scenario(scenario).tasks
    expected = "tensor:embedding_weights:rank=0:tp_rank=0:pp_rank=0"
    weight_tasks = [task for task in tasks if task.metadata.get("event_kind")
                    in {"embedding", "lm_head_projection"}
                    and task.metadata.get("memory_accesses")]
    assert {task.metadata["event_kind"] for task in weight_tasks} == {"embedding", "lm_head_projection"}
    observed = []
    for task in weight_tasks:
        metadata = task.metadata
        logical = "embedding_weights" if metadata["event_kind"] == "embedding" else "lm_head_weights"
        assert metadata["weight_tensor_id"] == logical
        assert metadata["tensor_id"] == logical
        assert metadata["weight_buffer_id"] == "embedding_weights"
        reads = [row for row in metadata["memory_accesses"]
                 if row.get("buffer_id") == expected and row["operation"] == "read"]
        assert reads
        assert all(row["allocation_generation"] == 0 and row["allocation_size_bytes"] == 4096
                   and row["allocation_arena"] == "resident" for row in reads)
        observed.extend(row for row in metadata["physical_allocations"]
                        if row["buffer_id"] == expected)
        if strategy == "HBF":
            assert ("hbf0.memory", expected, 0) in metadata["hbf_preloaded_buffers"]
    assert observed and {row["size_bytes"] for row in observed} == {4096}
    assert not any("lm_head_weights" in row["buffer_id"]
                   for task in tasks for row in task.metadata.get("physical_allocations", ()))


@pytest.mark.parametrize("field,value", [
    ("weight_buffer_id", "authored-head"), ("weight_allocation_id", "copy-head"),
    ("allocation_id", "explicit-allocation"), ("weight_alias_of", "explicit-root"),
    ("weight_offset_bytes", 0), ("weight_generation", 0),
    ("weight_owner_component_id", "hostmem0"), ("runtime_copy_of", "embedding_weights"),
    ("f16_weight_tensors", ()), ("physical_allocations", ()),
])
def test_explicit_physical_contracts_are_never_overwritten(field, value):
    scenario = _planned(small=True)
    plan = build_parallel_plan(scenario)
    assert _shared_input_output_weight_storage(scenario, plan, plan.ranks[0],
        "lm_head_weights", "gpu0", {field: value}) is None


@pytest.mark.parametrize("degree", ["tp_degree", "pp_degree", "ep_degree"])
def test_different_parallel_views_do_not_share_a_physical_buffer(degree):
    scenario = _planned(small=True)
    plan = build_parallel_plan(scenario)
    plan = replace(plan, **{degree: 2})
    assert _shared_input_output_weight_storage(scenario, plan, plan.ranks[0],
        "lm_head_weights", "gpu0") is None


def test_runtime_copies_and_different_owners_keep_independent_storage():
    scenario = _planned(small=True)
    plan = build_parallel_plan(scenario)
    rank = plan.ranks[0]
    control = scenario.placement.metadata["control_plane"]
    decision = control["decision"]
    details = {**decision["weight_tensor_details"], "lm_head_weights": {
        "runtime_copy_of": "embedding_weights", "residency": "resident_storage"}}
    copied = replace(scenario, placement=replace(scenario.placement,
        metadata={**scenario.placement.metadata, "control_plane": {
            **control, "decision": {**decision, "weight_tensor_details": details}}}))
    assert _shared_input_output_weight_storage(copied, plan, rank, "lm_head_weights", "gpu0") is None
    moved = replace(scenario, placement=replace(scenario.placement,
        tensor_to_component={**scenario.placement.tensor_to_component,
            "embedding_weights": "hbf0", "lm_head_weights": "hostmem0"}))
    assert _shared_input_output_weight_storage(moved, plan, rank, "lm_head_weights", "gpu0") is None


@pytest.mark.parametrize("change", ["layout", "dtype", "strides", "gguf"])
def test_unproven_storage_formats_do_not_alias(change):
    scenario = _planned(small=True)
    graph = scenario.model.graph
    if change == "gguf":
        graph = replace(graph, attributes={**graph.attributes, "metadata": {"gguf_output_binding": {}}})
    elif change == "strides":
        # Nested authoring dictionaries remain mutable despite frozen model
        # dataclasses. The physical qualification must see such an edit.
        plan = build_parallel_plan(scenario)
        next(tensor for tensor in graph.tensors
             if tensor.tensor_id == "lm_head_weights").attributes["strides"] = (2, 64)
        assert _shared_input_output_weight_storage(scenario, plan, plan.ranks[0],
            "lm_head_weights", "gpu0") is None
        return
    else:
        tensors = []
        for tensor in graph.tensors:
            if tensor.tensor_id == "lm_head_weights":
                tensor = replace(tensor, **({"layout": "packed"} if change == "layout" else
                    {"dtype": "fp16"}))
            tensors.append(tensor)
        # The earlier authoritative graph gate rejects a mismatched tensor
        # dtype/layout before physical storage binding can be reached.
        with pytest.raises(ValueError, match="lm_head"):
            replace(graph, tensors=tuple(tensors))
        return
    edited = replace(scenario, model=replace(scenario.model, graph=graph))
    plan = build_parallel_plan(scenario)
    assert _shared_input_output_weight_storage(edited, plan, plan.ranks[0], "lm_head_weights", "gpu0") is None
