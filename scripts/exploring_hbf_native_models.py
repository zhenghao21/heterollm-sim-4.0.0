"""Paper model inputs for the existing graph-native simulator.

This module only authors configuration. It does not execute operators, estimate
their costs, implement serving, or import the separate ``paper_*`` engine.
Unsupported architecture semantics remain unavailable rather than being
silently replaced by ordinary full attention.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.ir import LayerSpec, ModelSpec, build_model_graph_from_layer_specs
from heterollm_sim.model_presets import UnsupportedPresetError


_CATALOG = {
    "llama3_405b": {
        "name": "Llama 3.1 405B BF16（Meta 官方 KV8 几何）",
        "native_executable": True,
        "architecture": "llama",
        "layers": 126,
        "hidden_size": 16384,
        "intermediate_size": 53248,
        "attention_heads": 128,
        "kv_heads": 8,
        "head_dim": 128,
        "vocabulary_size": 128256,
        "max_sequence_length": 131072,
        "tie_word_embeddings": False,
        "weight_dtype": "bf16",
        "kv_dtype": "bf16",
        "sources": [
            "https://raw.githubusercontent.com/meta-llama/llama-models/main/models/sku_list.py",
            "https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3/model.py",
            "https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct",
        ],
        "source_scope": "Meta 405B bf16-mp8 SKU and its FFN rounding formula; no checkpoint weights downloaded.",
        "limitations": [
            "论文使用 Llama3 405B 名称；这里明确采用发布的 Llama3.1 405B 几何。",
            "独立官方 BF16 配置；不复用或改写同名 GGUF KV16 Q4_K_M 变体。",
            "通用原生图已执行 RMSNorm 算术，但尚未单独声明其仿射向量存储；矩阵容量不包含这些向量。",
            "输出 logits 明确使用 BF16；不能复用此前旁路执行器的 FP32 输出工作区假设。",
        ],
    },
    "llama4_maverick": {
        "name": "Llama 4 Maverick 17B 128E（文本骨干）",
        "native_executable": True,
        "architecture": "chunked_full_gqa_moe",
        "layers": 48,
        "hidden_size": 5120,
        "intermediate_size": 16384,
        "expert_intermediate_size": 8192,
        "attention_heads": 40,
        "kv_heads": 8,
        "head_dim": 128,
        "vocabulary_size": 202048,
        "max_sequence_length": 1048576,
        "tie_word_embeddings": False,
        "weight_dtype": "bf16",
        "kv_dtype": "bf16",
        "routed_experts": 128,
        "experts_per_token": 1,
        "shared_experts": 1,
        "moe_layer_step": 2,
        "full_attention_interval": 4,
        "attention_chunk_size": 8192,
        "sources": [
            "https://raw.githubusercontent.com/meta-pytorch/torchtune/main/torchtune/models/llama4/_model_builders.py",
            "https://raw.githubusercontent.com/meta-pytorch/torchtune/main/torchtune/models/llama4/_component_builders.py",
            "https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama4/moe.py",
            "https://raw.githubusercontent.com/huggingface/transformers/v4.51.3/src/transformers/models/llama4/modeling_llama4.py",
        ],
        "source_scope": "Named Meta Maverick builder and text component implementations; gated checkpoint not downloaded.",
        "attention_temperature_tuning": False,
        "unsupported_reasons": [],
        "limitations": [
            "文本骨干；不含视觉编码器及投影器。",
            "保留原生确定性均衡专家直方图；不冒充实测路由分布或独立随机期望触及专家数。",
            "固定块 KV 使用有限环形缓冲；同一请求 Prefill 调用必须在块边界分段。",
            "温度缩放能力已实现；默认遵循 Meta ModelArgs 的 False，论文未披露是否启用。",
            "原生通用 RMSNorm 算术存在，仿射权重向量尚未独立声明；其字节缺项单列。",
        ],
    },
}


def native_paper_model_catalog() -> dict[str, dict]:
    """Return independent inputs; mutating metadata cannot change later builds."""
    return deepcopy(_CATALOG)


def native_paper_model_metadata(model_id: str) -> dict:
    try:
        return deepcopy(_CATALOG[model_id])
    except KeyError as exc:
        raise ValueError("未知论文模型：" + str(model_id)) from exc


def build_native_paper_model(model_id: str) -> ModelSpec:
    """Create an ordinary ModelSpec, or reject a genuinely unsupported model."""
    spec = native_paper_model_metadata(model_id)
    if not spec["native_executable"]:
        raise UnsupportedPresetError("该模型尚不能由原生后端执行：" + "；".join(spec["unsupported_reasons"]))

    h = spec["hidden_size"]
    q = spec["attention_heads"] * spec["head_dim"]
    kv = spec["kv_heads"] * spec["head_dim"]
    attention_bytes = 2 * (2 * h * q + 2 * h * kv)
    embedding_bytes = 2 * h * spec["vocabulary_size"]
    missing_norm_bytes = 2 * h * (2 * spec["layers"] + 1)
    layers = []
    for index in range(spec["layers"]):
        moe = bool(spec.get("routed_experts") and (index + 1) % spec["moe_layer_step"] == 0)
        chunked = bool(spec.get("attention_chunk_size") and (index + 1) % spec["full_attention_interval"] != 0)
        intermediate = spec["expert_intermediate_size"] if moe else spec["intermediate_size"]
        ffn_parameters = 3 * h * intermediate
        if moe:
            ffn_parameters = (ffn_parameters * (spec["routed_experts"] + spec["shared_experts"])
                              + h * spec["routed_experts"])
        layer_metadata = {"source_configuration": model_id}
        if moe:
            layer_metadata["moe_routing"] = {
                "gating": "sigmoid", "normalize_selected_weights": False, "weight_scale": 1.0,
                "weight_application": "input", "selection_before_gating": True,
                "load_model": "deterministic_balanced_histogram",
            }
        layers.append(LayerSpec(
            layer_id="layer-{:03d}".format(index), kind="moe" if moe else "dense", hidden_size=h,
            intermediate_size=intermediate, attention_heads=spec["attention_heads"], kv_heads=spec["kv_heads"],
            attention_head_dim=spec["head_dim"], sequence_mixer="chunked_attention" if chunked else "full_attention",
            attention_chunk_size=spec["attention_chunk_size"] if chunked else 0,
            use_rope=(not spec.get("attention_chunk_size") or chunked),
            attention_temperature_tuning=(spec.get("attention_temperature_tuning", False) and not chunked),
            num_experts=spec["routed_experts"] if moe else 1, experts_per_token=1,
            shared_expert_intermediate_size=intermediate if moe else 0, shared_expert_gate=False,
            gated_mlp=True, dtype="bf16", weight_bytes=attention_bytes + 2 * ffn_parameters,
            metadata=layer_metadata,
        ))
    matrix_bytes = sum(layer.weight_bytes for layer in layers) + 2 * embedding_bytes
    metadata = {
        "experiment_model_id": model_id, "source": "official_configuration_native_graph",
        "configuration": spec,
        "resident_weight_basis": "BF16 matrix tensors; undeclared RMSNorm affine vectors excluded",
        "native_matrix_weight_bytes": matrix_bytes, "undeclared_rmsnorm_affine_bytes": missing_norm_bytes,
        "official_tensor_geometry_weight_bytes": matrix_bytes + missing_norm_bytes,
        "gated_checkpoint_downloaded": False, "paper_side_backend_used": False,
    }
    graph = build_model_graph_from_layer_specs(
        spec["name"], layers, architecture=spec["architecture"],
        vocabulary_size=spec["vocabulary_size"], max_sequence_length=spec["max_sequence_length"],
        embedding_weight_bytes=embedding_bytes, output_weight_bytes=embedding_bytes,
        tie_word_embeddings=spec["tie_word_embeddings"], output_head_dtype="bf16",
        metadata=metadata,
    )
    return ModelSpec(name=spec["name"], graph=graph, metadata=metadata)


__all__ = ["native_paper_model_catalog", "native_paper_model_metadata", "build_native_paper_model"]
