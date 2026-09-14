"""Join native activity evidence with simulator task metadata conservatively."""
from __future__ import annotations
from typing import Any, Mapping, Sequence

def task_stage(task: Any) -> str:
    meta = getattr(task, "metadata", {}) or {}
    explicit = str(meta.get("native_stage", "")).lower()
    if explicit: return explicit
    kind = " ".join(str(meta.get(key, "")).lower() for key in ("event_kind", "operator_id", "projection_id", "operator_class", "phase"))
    generic = {"kernel_launch", "model_weight_access", "operator_input_transfer", "operator_output_transfer", "gpu_gemm", "cpu_dispatch", "memory"}
    name = str(getattr(task, "name", "")).lower()
    if not kind or any(token in kind for token in generic): kind += " " + name
    if "logits_d2h" in kind or "host_output" in kind or "output.completion" in kind: return "output"
    if "qkv" in kind or "attention.qkv" in kind or "q_proj" in kind or "k_proj" in kind or "v_proj" in kind: return "attention_qkv"
    if "attention.output" in kind or "attn_out" in kind or "softmax" in kind or "attention_residual" in kind: return "attention_output"
    if "kv_read" in kind or "kv_append" in kind or "kv_cache" in kind or "cache_k" in kind or "cache_v" in kind: return "kv"
    if "lm_head" in kind or "logit" in kind: return "lm_head"
    if "ffn" in kind or "feed_forward" in kind or "mlp" in kind or "gate_proj" in kind or "up_proj" in kind or "down_proj" in kind or "swiglu" in kind or "silu" in kind: return "ffn"
    if "norm" in kind: return "normalization"
    return "unknown"

def map_native_events_to_tasks(events: Sequence[Mapping[str, Any]], tasks: Sequence[Any]) -> dict[str, Any]:
    """Return candidate matches; never invent a task id for ambiguous events."""
    indexed = [(getattr(t, "task_id", None), task_stage(t), getattr(t, "metadata", {}) or {}) for t in tasks]
    records=[]; candidate_records=[]; unmatched=[]; unsupported=[]
    for event in events:
        if event.get("kind", "kernel") != "kernel":
            unsupported.append({"event": dict(event), "status": "unsupported_kind"})
            continue
        stage = str(event.get("stage", "")).lower()
        candidates=[tid for tid,s,m in indexed if stage and s == stage]
        if event.get("task_id") and any(tid == event["task_id"] for tid,_,_ in indexed):
            candidates=[event["task_id"]]
        if len(candidates)==1:
            record={"event":dict(event),"task_id":candidates[0],"stage":stage,"confidence":1.0 if event.get("task_id") else 0.25,"status":"matched" if event.get("task_id") else "candidate"}
            (records if event.get("task_id") else candidate_records).append(record)
        else:
            unmatched.append({"event":dict(event),"candidate_task_ids":candidates,"status":"ambiguous" if candidates else "unmatched"})
    from collections import Counter
    return {"schema":"native-task-mapping/v1","matched":records,"candidates":candidate_records,"unmatched":unmatched,"unsupported":unsupported,"matched_count":len(records),"candidate_count":len(candidate_records),"unmatched_count":len(unmatched),"unsupported_count":len(unsupported),"stage_event_counts":dict(Counter(str(e.get("stage", "unknown")) for e in events if e.get("kind")=="kernel")),"unsupported_by_kind":dict(Counter(str(e.get("kind", "unknown")) for e in unsupported)),"status":"partial" if unmatched or candidate_records or unsupported else "complete"}

__all__=["task_stage","map_native_events_to_tasks"]
