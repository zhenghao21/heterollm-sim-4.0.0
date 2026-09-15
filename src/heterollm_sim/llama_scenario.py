"""Apply a llama.cpp runtime contract to an authored scenario.

This adapter is deliberately small: it projects llama.cpp's scheduler/KV and
offload settings onto the existing V4 scheduler, KV policy, and control-plane
placement contracts, then runs the normal placement planner so the resulting
scenario contains real CPU/GPU ownership evidence.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .config import ScenarioConfig
from .control_plane_planner import PlacementPolicy, plan_runtime_placement
from .ir import KVCachePolicy, PlacementSpec, SchedulerSpec, model_graph_execution_view
from .runtime_adapters import LlamaCppRuntimeConfig, LLAMA_HYBRID_BATCH_SCHEMA, LLAMA_SLOT_ORDER_SCHEMA


def _llama_mixed_batching_contract(
    scenario: ScenarioConfig, enabled: bool, *, config: LlamaCppRuntimeConfig | None = None,
    recurrent_contract: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Bound the locked server's decode-first, remaining-budget prompt fill.

    server-context.cpp adds generating tokens before pending prompt tokens
    under cont_batching.  Ordinary dense full-attention text uses the existing
    physical mixed-batch lowerer; this does not prove recurrent rectangular
    batches, expert dispatch, MTP or embedding/adapter compatibility.
    """
    view = model_graph_execution_view(scenario.model.graph, schema_version=scenario.model.schema_version)
    has_recurrent = any(x.layer.is_linear_attention for x in view.layer_instances)
    proof = recurrent_contract if isinstance(recurrent_contract, Mapping) else {}
    qualified_recurrent = bool(
        config is not None and config.kv_unified and config.parallel <= config.ubatch
        and proof.get("schema") == LLAMA_HYBRID_BATCH_SCHEMA and proof.get("status") == "source_derived"
        and proof.get("physical_lowering") == "equal_length_stateful_ubatches"
        and proof.get("kv_unified") is True and proof.get("recurrent_rollback_snapshots") == 0
        and proof.get("accuracy_validated") is False and proof.get("native_latency_used") is False
        and view.architecture in proof.get("architectures", [])
        and type(proof.get("captured_sequence_capacity")) is int
        and 1 <= config.parallel <= proof["captured_sequence_capacity"]
        and type(proof.get("captured_ubatch_capacity")) is int
        and config.ubatch <= proof["captured_ubatch_capacity"]
        and proof.get("source_sha256") and proof.get("runtime_log_ref", {}).get("sha256")
        and proof.get("capabilities", {}).get("supports_batched_stateful_execution") is True
        and proof.get("capabilities", {}).get("supports_equal_length_stateful_ubatches") is True)
    reason = "ordinary_dense_full_attention_text"
    if scenario.workload.mtp is not None or view.mtp_descriptors:
        reason = "mtp_batching_unproven"
    elif any(x.layer.is_moe or x.layer.kind != "dense" or x.layer.shared_expert_intermediate_size for x in view.layer_instances):
        reason = "expert_batching_unproven"
    elif has_recurrent and not qualified_recurrent:
        reason = "recurrent_physical_ubatch_unproven"
    elif not scenario.model.text_backbone_only or set(scenario.model.supported_modalities) != {"text"}:
        reason = "non_text_model_batching_unproven"
    else:
        request_metadata = [scenario.workload.metadata, *(request.metadata for request in scenario.workload.requests)]
        for metadata in request_metadata:
            raw = metadata.get("modalities", metadata.get("modality", ("text",)))
            modalities = (raw,) if isinstance(raw, str) else raw
            if (not isinstance(modalities, (tuple, list, set, frozenset))
                    or any(str(value).strip().lower() != "text" for value in modalities)
                    or any(metadata.get("has_" + name) for name in ("image", "audio", "video"))):
                reason = "non_text_request_batching_unproven"
                break
            if (any(metadata.get(key) for key in ("lora", "loras", "adapters", "input_embeddings", "need_embd", "embeddings"))
                    or metadata.get("task_type", "completion") not in ("completion", "generation")):
                reason = "request_task_or_adapter_compatibility_unproven"
                break
    qualified = reason == "ordinary_dense_full_attention_text"
    if qualified and has_recurrent:
        reason = "source_bound_hybrid_equal_length_ubatches"
    return {
        "schema": "llama.cpp.mixed-phase-batching/v1",
        "status": "enabled" if enabled and qualified else "disabled" if not enabled else "unsupported",
        "reason": reason if enabled else "cont_batching_disabled",
        "graph_qualified": qualified,
        "source": "locked llama.cpp tools/server/server-context.cpp:update_slots; can_batch_with; can_split",
        "source_rule": "generating rows first; append compatible ordinary prompt rows until n_batch; split by n_ubatch",
        "scope": "causal dense text; hybrid allowed only with a source-bound equal-length microbatch proof; no adapters, MTP or experts",
        "recurrent_source_contract": dict(proof) if qualified and has_recurrent else None,
        "evidence_kind": "source_derived_execution_semantics",
        "accuracy_validated": False,
    }



def _llama_slot_order_qualification(
    scenario: ScenarioConfig, config: LlamaCppRuntimeConfig,
    proof: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Qualify the closed workload, never infer native order from its latency."""
    requests = scenario.workload.requests
    scheduler = scenario.workload.scheduler
    previous = scenario.workload.metadata.get("llama_cpp_slot_order", {})
    previous = previous if isinstance(previous, Mapping) else {}
    authored_order = previous.get("previous_phase_candidate_order", scheduler.phase_candidate_order)
    if proof is None:
        return {"schema": "llama.cpp.slot-order-qualification/v1", "status": "not_requested",
                "qualified": False, "applied": False, "reasons": ["no_source_contract"],
                "phase_candidate_order": scheduler.phase_candidate_order,
                "previous_phase_candidate_order": authored_order, "accuracy_validated": False}
    reasons = []
    if not isinstance(proof, Mapping):
        reasons.append("slot_order_contract_must_be_mapping")
        proof = {}
    required = ("explicit_fresh_cohort", "same_arrival", "request_count_at_most_slots", "no_slot_reuse",
                "equal_priority_deadline", "no_preemption", "no_priority_aging", "compatible_causal_text")
    requirements = proof.get("requirements", {})
    sources = proof.get("source_sha256", {})
    if (proof.get("schema") != LLAMA_SLOT_ORDER_SCHEMA or proof.get("status") != "source_derived"
            or proof.get("phase_candidate_order") != "stable_admission"
            or proof.get("native_slot_iteration") != "vector_order"
            or proof.get("preserves_engine_start_definition") is not True
            or proof.get("native_latency_used") is not False or proof.get("accuracy_validated") is not False
            or not isinstance(requirements, Mapping) or any(requirements.get(key) is not True for key in required)
            or not isinstance(sources, Mapping) or not sources):
        reasons.append("slot_order_source_contract_unverified")
    if not requests:
        reasons.append("explicit_closed_request_set_required")
    if len(requests) > config.parallel:
        reasons.append("request_count_exceeds_fresh_slots")
    if requests and len({r.arrival_ns for r in requests}) != 1:
        reasons.append("dynamic_or_staggered_arrivals_unproven")
    if requests and len({(r.priority, r.deadline_ns) for r in requests}) != 1:
        reasons.append("priority_or_deadline_order_unproven")
    if scenario.workload.arrival_rate_rps != 0:
        reasons.append("arrival_stream_unproven")
    if scheduler.policy != "decode_first":
        reasons.append("priority_aging_unproven")
    if scheduler.preemption_enabled:
        reasons.append("preemption_or_slot_reuse_unproven")
    if scheduler.prefill_chunk_tokens < config.batch:
        reasons.append("prefill_chunk_shorter_than_native_batch")
    view = model_graph_execution_view(scenario.model.graph, schema_version=scenario.model.schema_version)
    if scenario.workload.mtp is not None or view.mtp_descriptors:
        reasons.append("mtp_slot_reuse_unproven")
    if not scenario.model.text_backbone_only or set(scenario.model.supported_modalities) != {"text"}:
        reasons.append("non_text_slot_compatibility_unproven")
    for metadata in [scenario.workload.metadata, *(r.metadata for r in requests)]:
        if (metadata.get("fresh_cohort") is False or metadata.get("initial_slots_empty") is False
                or any(metadata.get(key) for key in ("slot_reuse", "reuse_slots", "dynamic_requests",
                                                     "initial_slot_state", "resumed_slot_state", "resume_state"))):
            reasons.append("nonfresh_or_reused_slots_unproven")
        raw = metadata.get("modalities", metadata.get("modality", ("text",)))
        modalities = (raw,) if isinstance(raw, str) else raw
        if (not isinstance(modalities, (tuple, list, set, frozenset))
                or any(str(v).strip().lower() != "text" for v in modalities)
                or any(metadata.get(key) for key in ("lora", "loras", "adapters", "input_embeddings", "need_embd", "embeddings"))
                or metadata.get("task_type", "completion") not in ("completion", "generation")):
            reasons.append("slot_task_compatibility_unproven")
    qualified = not reasons
    order = "stable_admission" if qualified else (authored_order if previous.get("applied") else scheduler.phase_candidate_order)
    return {"schema": "llama.cpp.slot-order-qualification/v1", "status": "enabled" if qualified else "unsupported",
        "qualified": qualified, "applied": qualified, "reasons": list(dict.fromkeys(reasons)),
        "phase_candidate_order": order, "previous_phase_candidate_order": authored_order,
        "cohort": {"explicit_request_count": len(requests), "slot_capacity": config.parallel,
                   "arrival_ns": requests[0].arrival_ns if requests and len({r.arrival_ns for r in requests}) == 1 else None,
                   "request_ids": [r.request_id for r in requests],
                   "isolation_basis": "complete explicit workload with at most one initial request per slot; no arrival stream/preemption/reuse"},
        "source_contract": dict(proof), "scope": proof.get("scope"),
        "preserves_engine_start_definition": True, "accuracy_validated": False}


def apply_llama_runtime_config(
    scenario: ScenarioConfig,
    config: LlamaCppRuntimeConfig,
    *,
    materialize_placement: bool = True,
    recurrent_batching_contract: Mapping[str, Any] | None = None,
    slot_order_contract: Mapping[str, Any] | None = None,
) -> ScenarioConfig:
    """Return ``scenario`` with llama.cpp semantics lowered into typed fields."""
    if not isinstance(scenario, ScenarioConfig):
        raise TypeError("scenario must be a ScenarioConfig")
    if not isinstance(config, LlamaCppRuntimeConfig):
        raise TypeError("config must be a LlamaCppRuntimeConfig")
    if config.kv_type_k is not None and config.kv_type_v is not None and config.kv_type_k.casefold() != config.kv_type_v.casefold():
        raise ValueError("simulator currently requires identical llama.cpp K/V cache dtypes")
    for request in scenario.workload.requests:
        required = int(request.prompt_tokens) + int(request.output_tokens)
        if required > config.context:
            raise ValueError(
                "llama.cpp context {} is smaller than request {} token span {}"
                .format(config.context, request.request_id, required)
            )
    if not scenario.workload.requests:
        required = int(scenario.workload.prompt_tokens) + int(scenario.workload.output_tokens)
        if required > config.context:
            raise ValueError(
                "llama.cpp context {} is smaller than synthetic request token span {}"
                .format(config.context, required)
            )
    recurrent_proof = (recurrent_batching_contract if recurrent_batching_contract is not None
                       else scenario.workload.metadata.get("llama_cpp_recurrent_batching_contract"))
    mixed_batching = _llama_mixed_batching_contract(scenario, config.cont_batching,
                                                  config=config, recurrent_contract=recurrent_proof)
    hybrid_capabilities = ({
        "supports_batched_stateful_execution": False,
        "supports_equal_length_stateful_ubatches": False,
    } if recurrent_proof is not None else {})
    if mixed_batching.get("recurrent_source_contract") is not None:
        hybrid_capabilities = {
            "llama_cpp_recurrent_batching_contract": dict(recurrent_proof),
            "supports_batched_stateful_execution": True,
            "supports_equal_length_stateful_ubatches": True,
        }
    slot_proof = (slot_order_contract if slot_order_contract is not None
                  else scenario.workload.metadata.get("llama_cpp_slot_order_contract"))
    slot_order = _llama_slot_order_qualification(scenario, config, slot_proof)
    slot_metadata = {"llama_cpp_slot_order": slot_order}
    if isinstance(slot_proof, Mapping):
        slot_metadata["llama_cpp_slot_order_contract"] = dict(slot_proof)
    scheduler = scenario.workload.scheduler
    scheduler = replace(
        scheduler,
        max_num_seqs=config.parallel,
        max_num_batched_tokens=config.batch,
        max_num_ubatch_tokens=config.ubatch,
        mode="continuous" if config.cont_batching else "static",
        mixed_phase_batching=mixed_batching["status"] == "enabled",
        phase_candidate_order=slot_order["phase_candidate_order"],
    )
    workload = replace(
        scenario.workload,
        scheduler=scheduler,
        metadata={
            **scenario.workload.metadata,
            **hybrid_capabilities,
            **slot_metadata,
            "llama_cpp_runtime": config.to_dict(),
            "llama_cpp_mixed_phase_batching": mixed_batching,
            "context_limit_semantics": "per_slot_runtime_limit",
        },
    )
    kv = scenario.placement.kv_policy
    # llama.cpp's context is a per-slot reservation; preserve explicit KV
    # placement/offload targets while changing the slot geometry and dtype.
    kv_cache_component = kv.cache_component
    if not config.offload_kqv:
        host_memory = next(
            (component.component_id for component in scenario.hardware.components
             if component.normalized_kind in {"host_memory", "dram", "ddr", "ddr_memory"}),
            None,
        )
        if host_memory is None:
            raise ValueError("llama.cpp --no-kv-offload requires a host memory component")
        kv_cache_component = host_memory
    kv = replace(
        kv,
        cache_component=kv_cache_component,
        tokens_per_page=max(1, kv.tokens_per_page),
        dtype=config.kv_type_k or config.kv_type_v or kv.dtype,
    )
    placement_metadata = dict(scenario.placement.metadata)
    control = dict(placement_metadata.get("control_plane", {}))
    policy = dict(control.get("policy", {}))
    options = dict(policy.get("options", {}))
    options.update({
        # This llama.cpp build counts the output layer in -ngl: -ngl=12
        # offloads the output layer plus the last 11 repeating blocks.
        "gpu_loadable_layers": scenario.model.num_layers + 1 if config.gpu_layers < 0 else config.gpu_layers,
        "gpu_loadable_order": "tail",
        # llama.cpp keeps token_embd.weight CPU-resident when the output
        # matrix is a logical tied alias, then materializes a GPU runtime copy
        # for lm_head.  This avoids pulling the full embedding tensor over
        # PCIe on every decode token.
        "tied_weight_runtime_copies": config.gpu_layers != 0 and scenario.model.output_weight_bytes <= 0,
    })
    policy["options"] = options
    control["policy"] = policy
    evidence = dict(control.get("evidence", {}))
    evidence["llama_cpp_runtime"] = config.to_dict()
    evidence["llama_cpp_runtime_fingerprint"] = config.fingerprint
    control["evidence"] = evidence
    placement_metadata["control_plane"] = control
    placement_metadata["llama_cpp_runtime"] = config.to_dict()
    placement_metadata["llama_cpp_runtime_fingerprint"] = config.fingerprint
    placement = replace(scenario.placement, kv_policy=kv, metadata=placement_metadata)
    lowered = replace(
        scenario,
        placement=placement,
        workload=workload,
        llama_cpp_config=config,
        fusion_policy=replace(scenario.fusion_policy, flash_attention=config.flash_attn),
        assumptions=tuple(dict.fromkeys((*scenario.assumptions, "llama.cpp runtime config lowered into typed scheduler/KV/offload semantics"))),
    )
    if materialize_placement:
        decision = plan_runtime_placement(
            lowered,
            PlacementPolicy(
                gpu_loadable_layers=options["gpu_loadable_layers"],
                gpu_loadable_order=options["gpu_loadable_order"],
                tied_weight_runtime_copies=options["tied_weight_runtime_copies"],
            ),
        )
        lowered = decision.apply(lowered)
        # Keep the runtime evidence alongside planner evidence after apply().
        metadata = dict(lowered.placement.metadata)
        cp = dict(metadata.get("control_plane", {}))
        ev = dict(cp.get("evidence", {}))
        ev.update({"llama_cpp_runtime": config.to_dict(), "llama_cpp_runtime_fingerprint": config.fingerprint})
        cp["evidence"] = ev
        lowered = replace(lowered, placement=replace(lowered.placement, metadata={**metadata, "control_plane": cp}))
    return lowered


__all__ = ["apply_llama_runtime_config"]
