"""Match source CUDA dispatch ownership to modeled operations, never by count.

Names here are the pinned llama graph builder's semantic labels and physical
GGUF weight names. Unknown operations and mismatched fusion boundaries fail;
no proportional split of a physical kernel/DRAM body is made.
"""
from __future__ import annotations

from collections import defaultdict
import re


def _layer(name):
    match = re.fullmatch(r"layer-(\d+)", name or "")
    if not match:
        raise ValueError("CUDA dispatch task lacks its explicit layer identity")
    return int(match[1])


def source_operation_keys(node):
    """A semantic key can span several GGML nodes, e.g. RMS + learned scale."""
    output = node["output"]
    name, op = output["name"], output["op"]
    inputs = {item["slot"]: item["tensor"] for item in node["sources"]}
    if op in {"VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE", "NONE"}:
        return frozenset()
    if op == "MUL_MAT":
        weight = inputs.get(0, {}).get("name", "")
        if weight.endswith(".weight"):
            return frozenset({("weight", weight)})
        for prefix, role in (("kq", "attention_qk"), ("kqv", "attention_pv")):
            match = re.fullmatch(prefix + r"-(\d+)", name)
            if match:
                return frozenset({(int(match[1]), role)})
    if op in {"RMS_NORM", "MUL"}:
        # The scale's physical GGUF name is unambiguous even when several
        # intermediate GGML tensors are all named norm-<layer>.
        if op == "MUL":
            weight = inputs.get(1, {}).get("name", "")
            match = re.fullmatch(r"blk\.(\d+)\.(attn_norm|attn_q_norm|attn_k_norm|ffn_norm)\.weight", weight)
            if match:
                role = {"attn_norm": "input_norm", "attn_q_norm": "q_norm",
                        "attn_k_norm": "k_norm", "ffn_norm": "post_attention_norm"}[match[2]]
                return frozenset({(int(match[1]), role)})
            if weight == "output_norm.weight":
                return frozenset({("output", "norm")})
        elif name == "norm" and re.fullmatch(r"l_out-\d+", inputs.get(0, {}).get("name", "")):
            return frozenset({("output", "norm")})
        else:
            match = re.fullmatch(r"norm-(\d+)", name)
            if match:
                parent = inputs.get(0, {}).get("name", "")
                layer = int(match[1])
                parents = {f"Qcur-{layer}": "q_norm", f"Kcur-{layer}": "k_norm",
                           f"ffn_inp-{layer}": "post_attention_norm"}
                role = parents.get(parent)
                if (parent == f"l_out-{layer - 1}" or
                        (layer == 0 and re.fullmatch(r"CUDA\d+#embd#\d+", parent))):
                    role = "input_norm"
                if role is not None:
                    return frozenset({(layer, role)})
    if op == "ROPE":
        match = re.fullmatch(r"([QK])cur-(\d+)", name)
        if match:
            return frozenset({(int(match[2]), "rope_" + match[1].lower())})
    if op == "SET_ROWS":
        match = re.match(r"cache_([kv])_l(\d+)(?: |$)", name)
        if match:
            return frozenset({(int(match[2]), "set_rows_" + match[1])})
    for expected, prefix, role in (("SOFT_MAX", "kq_soft_max", "softmax"),
            ("CONT", "kqv_out", "attention_context_contiguous"),
            ("GLU", "ffn_swiglu", "mlp_activation"),
            ("ADD", "ffn_inp", "attention_residual"), ("ADD", "l_out", "mlp_residual")):
        match = re.fullmatch(prefix + r"-(\d+)", name)
        if op == expected and match:
            return frozenset({(int(match[1]), role)})
    if op == "ADD":
        for tensor in inputs.values():
            match = re.fullmatch(r"blk\.(\d+)\.attn_([qkv])\.bias", tensor["name"])
            if match:
                return frozenset({(int(match[1]), "attention_" + match[2] + "_bias")})
    raise ValueError(f"unmapped source CUDA operation: {op} {name}")


def modeled_operation_keys(task):
    meta = task.metadata
    event, projection = meta.get("event_kind"), meta.get("projection_id")
    name = meta.get("op_name", task.name)
    if meta.get("phase") != "kernel_launch" and event != "attention_context_contiguous":
        return None
    if projection:
        layer = _layer(meta["layer_id"]) if meta.get("layer_id") is not None else None
        if layer is not None or projection == "lm_head":
            weight = meta.get("weight_buffer_id")
            if weight:
                return frozenset({("weight", weight)})
            weights = meta.get("f16_weight_tensors", ())
            if weights:
                result = {("weight", item["name"]) for item in weights}
                if projection == "mlp.up_gate" and meta.get("fusion_enabled"):
                    result.add((layer, "mlp_activation"))
                return frozenset(result)
            segments = meta.get("projection_segments", ())
            if segments and all(segment.get("physical_tensor_name") for segment in segments):
                return frozenset(("weight", segment["physical_tensor_name"]) for segment in segments)
            raise ValueError("modeled CUDA projection lacks complete physical weight ownership: " + name)
    if meta.get("layer_id") is not None:
        layer = _layer(meta["layer_id"])
        roles = {"input_norm_reduce": "input_norm", "input_norm_apply": "input_norm",
                 "attention_q_norm_reduce": "q_norm", "attention_q_norm_apply": "q_norm",
                 "attention_k_norm_reduce": "k_norm", "attention_k_norm_apply": "k_norm",
                 "post_attention_norm_reduce": "post_attention_norm",
                 "post_attention_norm_apply": "post_attention_norm",
                 "softmax_reduce": "softmax", "softmax_normalize": "softmax"}
        if event in roles:
            return frozenset({(layer, roles[event])})
        if event in {"attention_residual", "mlp_residual", "mlp_activation",
                     "attention_context_contiguous", "attention_q_bias", "attention_k_bias", "attention_v_bias"}:
            return frozenset({(layer, event)})
        if event == "rope" and meta.get("rope_operand") in {"q", "k"}:
            return frozenset({(layer, "rope_" + meta["rope_operand"])})
        if event == "kv_native_set_rows":
            match = re.search(r"\.([kv])_set_rows$", name)
            if match:
                return frozenset({(layer, "set_rows_" + match[1])})
        if event == "output_row_selection":
            stage = meta.get("final_layer_output_selection", {}).get("stage")
            if stage in {"attention_output_rows", "residual_input_rows"}:
                return frozenset({(layer, stage)})
        for role in ("attention_qk", "attention_pv"):
            if name.endswith("." + role):
                return frozenset({(layer, role)})
    if event in {"final_norm_reduce", "final_norm_apply", "final_norm_rms_mul"}:
        return frozenset({("output", "norm")})
    raise ValueError("unmapped modeled CUDA operation: " + name)


def _row_selection_keys(nodes):
    """Resolve unnamed GET_ROWS through actual tensor producer/consumer links."""
    by_tensor = {}
    for node in nodes:
        identity = node["output"]["tensor"]
        if identity in by_tensor:
            raise ValueError("source tensor has more than one dispatch owner")
        by_tensor[identity] = node
    result = {}
    for node in nodes:
        if node["output"]["op"] != "GET_ROWS":
            continue
        identity = node["output"]["tensor"]
        users = [consumer for consumer in nodes
                 if consumer["output"]["op"] == "ADD"
                 and re.fullmatch(r"ffn_inp-\d+", consumer["output"]["name"])
                 and any(source["tensor"]["tensor"] == identity for source in consumer["sources"])]
        if len(users) != 1:
            raise ValueError("source GET_ROWS has no unique final attention residual consumer")
        layer = int(users[0]["output"]["name"].removeprefix("ffn_inp-"))
        inputs = {source["slot"]: source["tensor"] for source in node["sources"]}
        original = inputs.get(0)
        if original is None or inputs.get(1, {}).get("type") != "i32":
            raise ValueError("source GET_ROWS requires explicit data and I32 index inputs")
        producer = by_tensor.get(original["tensor"])
        if producer is not None and producer["output"]["op"] == "MUL_MAT":
            weights = {source["slot"]: source["tensor"] for source in producer["sources"]}
            if weights.get(0, {}).get("name") != f"blk.{layer}.attn_output.weight":
                raise ValueError("source GET_ROWS matrix input is not this layer's physical attention output")
            role = "attention_output_rows"
        elif (original["name"] == f"l_out-{layer - 1}" or
                (layer == 0 and re.fullmatch(r"CUDA\d+#embd#\d+", original["name"]))):
            role = "residual_input_rows"
        else:
            raise ValueError("source GET_ROWS has an unsupported physical input producer")
        result[identity] = frozenset({(layer, role)})
    return result


def match_cuda_dispatch_tasks(tasks, source_dispatches):
    """Return exact memberships; reject one modeled body crossing dispatches.

    This only discovers ownership. It deliberately changes neither physical
    costs nor dependencies. A caller must explicitly model device scheduling.
    """
    owners = {}
    source_keys = []
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("modeled CUDA task identities must be unique")
    if len({group["dispatch_index"] for group in source_dispatches}) != len(source_dispatches):
        raise ValueError("source CUDA dispatch identities must be unique")
    node_ids = [identity for group in source_dispatches for identity in group["node_ids"]]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("source CUDA nodes must have unique dispatch owners")
    source_nodes = [node for group in source_dispatches for node in group["source_nodes"]]
    row_keys = _row_selection_keys(source_nodes)
    for index, group in enumerate(source_dispatches):
        if (type(group["node_count"]) is not int or group["node_count"] < 0
                or len(group["node_ids"]) != group["node_count"]):
            raise ValueError("source CUDA dispatch node ownership is incomplete")
        keys = frozenset().union(*(row_keys[node["output"]["tensor"]]
            if node["output"]["op"] == "GET_ROWS" else source_operation_keys(node)
            for node in group["source_nodes"]))
        if group["node_count"] and not keys:
            raise ValueError("CUDA dispatch emits nodes without a modeled semantic operation")
        if not group["node_count"] and keys:
            raise ValueError("source operation emits no CUDA nodes; physical lowering needs an explicit no-op contract")
        for key in keys:
            if key in owners:
                raise ValueError("semantic operation crosses separate CUDA dispatches: " + str(key))
            owners[key] = index
        source_keys.append(keys)
    matched, covered = defaultdict(list), set()
    for task in tasks:
        keys = modeled_operation_keys(task)
        if keys is None:
            continue
        if not keys <= owners.keys():
            raise ValueError("modeled CUDA operation has no source dispatch: " + str(keys - owners.keys()))
        indices = {owners[key] for key in keys}
        if len(indices) != 1:
            raise ValueError("modeled CUDA body crosses native dispatch boundaries; split physical lowering: " + task.name)
        index = next(iter(indices))
        matched[index].append(task.task_id)
        covered.update(keys)
    if covered != owners.keys():
        raise ValueError("source CUDA operations lack physical modeled bodies: " + str(owners.keys() - covered))
    return tuple({"dispatch_index": group["dispatch_index"],
                  "first_node_ordinal": group["first_node_ordinal"],
                  "node_count": group["node_count"], "node_ids": group["node_ids"],
                  "launch_task_ids": tuple(matched[index])}
                 for index, group in enumerate(source_dispatches))
