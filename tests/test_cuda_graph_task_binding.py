"""Ownership matching must not guess weights, fusion, timings or task order."""
from copy import deepcopy
from dataclasses import replace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.cuda_graph_task_binding import (
    match_cuda_dispatch_tasks, modeled_operation_keys, source_operation_keys,
)


def tensor(identity, name, op="NONE", dtype="f32"):
    return {"tensor": identity, "name": name, "op": op, "type": dtype}


def node(identity, name, op, *inputs):
    return {"output": tensor(identity, name, op), "sources": [
        {"slot": index, "tensor": value} for index, value in enumerate(inputs)]}


def group(index, nodes, count=1):
    return {"dispatch_index": index, "source_nodes": nodes,
            "node_count": count, "node_ids": tuple(f"cuda-{index}-{j}" for j in range(count)),
            "first_node_ordinal": index}


def task(identity, **metadata):
    return TaskSpec(identity, "request", identity, TaskCategory.COMPUTE,
        dependencies=(), demands=(ResourceDemand("gpu.frontend", 1000),),
        metadata={"phase": "kernel_launch", "layer_id": "layer-000", **metadata})


def matrix(identity, weight):
    return node(identity, "node_" + identity, "MUL_MAT", tensor("w-" + weight, weight), tensor("input", "input"))


def test_many_modeled_norm_events_bind_one_native_kernel_without_modification():
    rms = node("rms", "norm-0", "RMS_NORM", tensor("input", "CUDA0#embd#0"))
    mul = node("mul", "attn_norm-0", "MUL", rms["output"], tensor("weight", "blk.0.attn_norm.weight"))
    source = (group(0, [rms, mul]),)
    tasks = (task("reduce", event_kind="input_norm_reduce"), task("apply", event_kind="input_norm_apply"))
    before = deepcopy((tasks, source))
    result = match_cuda_dispatch_tasks(tasks, source)
    assert result[0]["launch_task_ids"] == ("reduce", "apply")
    assert result[0]["node_count"] == 1
    assert (tasks, source) == before


def test_aggregated_physical_qkv_crossing_source_dispatches_is_rejected():
    source = (group(0, [matrix("q", "blk.0.attn_q.weight")]),
              group(1, [matrix("k", "blk.0.attn_k.weight")]))
    aggregate = task("qkv", projection_id="attention.qkv", projection_segments=(
        {"physical_tensor_name": "blk.0.attn_q.weight"}, {"physical_tensor_name": "blk.0.attn_k.weight"}))
    with pytest.raises(ValueError, match="crosses native dispatch"):
        match_cuda_dispatch_tasks((aggregate,), source)


def test_actual_f16_gate_up_activation_fusion_binds_to_one_source_group():
    source = (group(0, [matrix("gate", "blk.0.ffn_gate.weight"),
        matrix("up", "blk.0.ffn_up.weight"), node("act", "ffn_swiglu-0", "GLU")]),)
    fused = task("ffn", projection_id="mlp.up_gate", fusion_enabled=True,
        f16_weight_tensors=({"name": "blk.0.ffn_gate.weight"}, {"name": "blk.0.ffn_up.weight"}))
    assert match_cuda_dispatch_tasks((fused,), source)[0]["launch_task_ids"] == ("ffn",)


def test_missing_physical_weights_cannot_be_rebuilt_from_a_projection_label():
    with pytest.raises(ValueError, match="complete physical weight ownership"):
        modeled_operation_keys(task("output", projection_id="attention.output"))


def test_unknown_norm_parent_does_not_become_input_norm():
    with pytest.raises(ValueError, match="unmapped source"):
        source_operation_keys(node("rms", "norm-0", "RMS_NORM", tensor("input", "unknown-state")))


def test_tied_embedding_head_uses_the_actual_gguf_tensor_identity():
    source = (group(0, [matrix("head", "token_embd.weight")]),)
    head = task("head", layer_id=None, projection_id="lm_head", projection_segments=(
        {"physical_tensor_name": "token_embd.weight"},))
    assert match_cuda_dispatch_tasks((head,), source)[0]["launch_task_ids"] == ("head",)
    with pytest.raises(ValueError, match="no source dispatch"):
        match_cuda_dispatch_tasks((replace(head, metadata={**head.metadata,
            "projection_segments": ({"physical_tensor_name": "output.weight"},)}),), source)


def row_selection_case():
    wo = matrix("wo", "blk.1.attn_output.weight")
    ids = tensor("ids", "arbitrary-ids", dtype="i32")
    gather_a = node("ga", "node_900", "GET_ROWS", wo["output"], ids)
    gather_r = node("gr", "node_901", "GET_ROWS", tensor("previous", "l_out-0", "ADD"), ids)
    add = node("add", "ffn_inp-1", "ADD", gather_a["output"], gather_r["output"])
    source = tuple(group(i, [value]) for i, value in enumerate([wo, gather_a, gather_r, add]))
    tasks = (task("wo", layer_id="layer-001", projection_id="attention.output", weight_buffer_id="blk.1.attn_output.weight"),
        task("ga", layer_id="layer-001", event_kind="output_row_selection", final_layer_output_selection={"stage": "attention_output_rows"}),
        task("gr", layer_id="layer-001", event_kind="output_row_selection", final_layer_output_selection={"stage": "residual_input_rows"}),
        task("add", layer_id="layer-001", event_kind="attention_residual"))
    return tasks, source


def test_unnamed_get_rows_uses_tensor_links_and_real_weight_instead_of_name_numbers():
    tasks, source = row_selection_case()
    result = match_cuda_dispatch_tasks(tasks, source)
    assert [row["launch_task_ids"] for row in result] == [("wo",), ("ga",), ("gr",), ("add",)]


@pytest.mark.parametrize("mutation", ["index_type", "producer_weight", "consumer", "residual_parent"])
def test_get_rows_missing_or_ambiguous_source_semantics_are_rejected(mutation):
    tasks, source = row_selection_case()
    source = deepcopy(source)
    if mutation == "index_type": source[1]["source_nodes"][0]["sources"][1]["tensor"]["type"] = "f32"
    elif mutation == "producer_weight": source[0]["source_nodes"][0]["sources"][0]["tensor"]["name"] = "blk.1.ffn_down.weight"
    elif mutation == "consumer": source[3]["source_nodes"][0]["output"]["name"] = "unrelated"
    else: source[2]["source_nodes"][0]["sources"][0]["tensor"]["name"] = "unknown-state"
    with pytest.raises(ValueError, match="GET_ROWS"):
        match_cuda_dispatch_tasks(tasks, source)


@pytest.mark.parametrize("mutation", ["duplicate_task", "duplicate_dispatch", "duplicate_node", "zero_nodes", "unowned_operation"])
def test_incomplete_or_duplicate_ownership_cannot_pass(mutation):
    tasks, source = row_selection_case()
    source = deepcopy(source)
    if mutation == "duplicate_task": tasks = (*tasks, tasks[0])
    elif mutation == "duplicate_dispatch": source[1]["dispatch_index"] = source[0]["dispatch_index"]
    elif mutation == "duplicate_node": source[1]["node_ids"] = source[0]["node_ids"]
    elif mutation == "zero_nodes": source[0]["node_count"] = 0; source[0]["node_ids"] = ()
    else: tasks = tasks[:-1]
    with pytest.raises(ValueError):
        match_cuda_dispatch_tasks(tasks, source)
