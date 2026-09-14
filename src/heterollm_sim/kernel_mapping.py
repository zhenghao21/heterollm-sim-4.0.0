"""Conservative mapping of Nsight kernel evidence to simulator stages."""
from __future__ import annotations
from dataclasses import dataclass
import csv, json
from pathlib import Path
from typing import Any, Iterable, Mapping

STAGES = ("attention_qkv", "attention_output", "ffn", "kv", "lm_head", "output", "normalization", "quantize", "linear_attention_aux", "unknown")

@dataclass(frozen=True)
class KernelClassification:
    stage: str; confidence: float; reason: str

@dataclass(frozen=True)
class KernelMapping:
    name: str; stage: str; confidence: float; reason: str
    total_ns: float = 0.0; instances: int = 0; avg_ns: float = 0.0; median_ns: float | None = None; source: str | None = None
    def to_dict(self):
        return {"name": self.name, "stage": self.stage, "confidence": self.confidence, "reason": self.reason, "total_ns": self.total_ns, "instances": self.instances, "avg_ns": self.avg_ns, "median_ns": self.median_ns, **({"source": self.source} if self.source else {})}

def _num(row, keys, default=0.0):
    for key in keys:
        if row.get(key) not in (None, ""):
            try: return float(str(row[key]).replace(",", ""))
            except (TypeError, ValueError): pass
    return default

def classify_kernel(name: str, row: Mapping[str, Any] | None = None) -> KernelClassification:
    low = str(name or "").lower()
    if row:
        for key in ("stage", "graph_stage", "node_stage", "op_stage"):
            hint = str(row.get(key, "")).strip().lower()
            # An explicit ``unknown`` value is an absence of evidence.  Keep
            # looking for a semantic owner attached by the trace extractor.
            if hint in STAGES and hint != "unknown": return KernelClassification(hint, 1.0, f"explicit {key} annotation")
        # Semantic NVTX scopes carry the graph invocation identity even when
        # CUDA's demangled kernel name is the generic ``mul_mat_q`` family.
        # This mapping is deliberately conservative: only stable owner names
        # emitted by llama.cpp are accepted; arbitrary node_N ids stay unknown.
        owner = str(row.get("semantic_operator_id") or row.get("semantic_operator") or row.get("operator_id") or "").strip().lower()
        if owner:
            owner = owner.split("|", 1)[0]
            owner_name = owner.split(":", 1)[-1]
            if any(token in owner_name for token in ("qcur", "kcur", "vcur", "q_proj", "k_proj", "v_proj", "qkv")):
                return KernelClassification("attention_qkv", .90, "semantic operator owner marker")
            if any(token in owner_name for token in ("ffn_inp", "ffn_gate", "ffn_up", "ffn_down", "ffn_out", "gate_proj", "up_proj", "down_proj")):
                return KernelClassification("ffn", .90, "semantic operator owner marker")
            if any(token in owner_name for token in ("cache_k", "cache_v", "kv_cache", "k_cache", "v_cache")):
                return KernelClassification("kv", .90, "semantic operator owner marker")
            if any(token in owner_name for token in ("result_output", "lm_head", "output_head", "logits")):
                return KernelClassification("lm_head", .90, "semantic operator owner marker")
            if any(token in owner_name for token in ("norm", "rms_norm", "layer_norm")):
                return KernelClassification("normalization", .90, "semantic operator owner marker")
            if any(token in owner_name for token in (
                "attn_pregate", "gate_reshaped", "attn_residual", "gate_sigmoid",
            )):
                return KernelClassification("attention_output", .88, "semantic attention post-processing owner marker")
            if any(token in owner_name for token in ("kq", "kqv", "soft_max", "softmax", "attn_out", "out_proj")):
                return KernelClassification("attention_output", .88, "semantic operator owner marker")
            # Qwen3.5's gated-delta-net path emits explicit non-GEMM owners.
            # Keep these costs separate from QKV/FFN and from full-attention
            # post-processing; the planner can then calibrate the auxiliary
            # linear-mixer primitives without inventing a global multiplier.
            if any(token in owner for token in (
                "ssm_conv", "gated_delta_net", "q_conv_predelta", "k_conv_predelta",
                "conv_input", "conv_state_update", "cache_r_l", "cache_s_l",
                "a_softplus", "beta_sigmoid", "l_out",
            )):
                return KernelClassification("linear_attention_aux", .90, "semantic linear-attention auxiliary owner marker")
    rules = (("quantize", ("quantize", "dequant", "quant_"), .98), ("normalization", ("rms_norm", "layer_norm", "norm_f32", "norm_"), .98), ("kv", ("kv_cache", "cache_k", "cache_v", "flash_attn"), .86), ("attention_qkv", ("q_proj", "k_proj", "v_proj", "qkv", "query", "key", "value", "rope"), .91), ("attention_output", ("attn_out", "out_proj", "softmax", "context"), .90), ("ffn", ("ffn", "gate_proj", "up_proj", "down_proj", "swiglu", "silu"), .92), ("lm_head", ("lm_head", "output_head", "logits", "get_rows"), .92), ("output", ("device-to-host", "d2h", "host_output", "sampling"), .94))
    for stage, markers, confidence in rules:
        if any(marker in low for marker in markers): return KernelClassification(stage, confidence, "kernel name marker")
    return KernelClassification("unknown", 0.0, "no stage-specific kernel marker")

def map_kernel_row(row: Mapping[str, Any], *, source: str | None = None) -> KernelMapping:
    name = str(next((row.get(k) for k in ("Name", "name", "kernel", "kernel_name") if row.get(k) is not None), ""))
    classification = classify_kernel(name, row); stage, confidence, reason = classification.stage, classification.confidence, classification.reason
    total = _num(row, ("Total Time (ns)", "total_ns", "duration_ns")); instances = int(round(_num(row, ("Instances", "Num Calls", "instances"), 1.0))); avg = _num(row, ("Avg (ns)", "avg_ns"), total / instances if instances else 0.0); median = _num(row, ("Med (ns)", "median_ns"), 0.0)
    return KernelMapping(name, stage, confidence, reason, total, instances, avg, median or None, source)

def aggregate_kernel_mappings(items: Iterable[KernelMapping]):
    out = {s: {"stage": s, "kernel_count": 0, "instances": 0, "total_ns": 0.0, "weighted_avg_ns": 0.0, "confidence_min": 0.0, "confidence_mean": 0.0, "share": 0.0} for s in STAGES}
    for item in items:
        b = out[item.stage]; b["kernel_count"] += 1; b["instances"] += item.instances; b["total_ns"] += item.total_ns; b["weighted_avg_ns"] += item.avg_ns * item.instances; b["confidence_min"] = item.confidence if b["kernel_count"] == 1 else min(b["confidence_min"], item.confidence); b["confidence_mean"] += item.confidence
    total = sum(b["total_ns"] for b in out.values())
    for b in out.values():
        if b["instances"]: b["weighted_avg_ns"] /= b["instances"]
        if b["kernel_count"]: b["confidence_mean"] /= b["kernel_count"]
        b["share"] = b["total_ns"] / total if total else 0.0
    return out

def map_kernel_profile(rows, *, source=None):
    records = [map_kernel_row(r, source=source) for r in rows]
    return {"schema": "native-kernel-mapping/v1", "records": [r.to_dict() for r in records], "aggregate": aggregate_kernel_mappings(records)}

def load_kernel_csv(path):
    lines = [x for x in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines() if x.strip()]; start = next((i for i, x in enumerate(lines) if "," in x), 0); return list(csv.DictReader(lines[start:])) if lines else []

def load_kernel_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list): return data
    if isinstance(data, Mapping):
        for key in ("rows", "kernels", "records"):
            if isinstance(data.get(key), list): return data[key]
    raise ValueError("kernel JSON must be a list or contain rows/kernels/records")

__all__ = ["STAGES", "KernelClassification", "KernelMapping", "classify_kernel", "map_kernel_row", "aggregate_kernel_mappings", "map_kernel_profile", "load_kernel_csv", "load_kernel_json"]
