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
from .runtime_adapters import (
    LlamaCppRuntimeConfig,
    LLAMA_HYBRID_BATCH_SCHEMA,
    LLAMA_SLOT_ORDER_SCHEMA,
    normalize_llama_runtime_identity,
)
from .parallel import build_parallel_plan


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
    # Qualification describes the authored cohort as evidence.  The resolved
    # scheduler may still lower native defaults, but an explicit authored
    # aging/preemption/chunk override must remain visible as an unproven
    # source contract instead of being silently erased.
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


def _resolve_llama_scheduler_policy(
    scenario: ScenarioConfig,
    config: LlamaCppRuntimeConfig,
    slot_order: Mapping[str, Any],
    mixed_batching: Mapping[str, Any],
) -> tuple[SchedulerSpec, Mapping[str, Any]]:
    """Resolve the complete serving policy once, without inherited fields.

    The native server's prompt fill is bounded by its logical ``n_batch``
    budget; the physical ``n_ubatch`` split is applied later by the graph
    lowerer.  Keeping those two limits separate prevents a microbatch size
    from silently becoming a second scheduling round.
    """
    authored = scenario.workload.scheduler
    profile = config.policy
    if profile == "llama_cpp":
        if config.preemption_enabled:
            raise ValueError(
                "llama.cpp alignment rejects preemption_enabled=True; "
                "use policy='generic' or policy='experimental' for that behavior"
            )
        policy = "decode_first"
        preemption_enabled = False
        prefill_chunk_tokens = config.prefill_chunk_tokens or config.batch
        source = (
            "source_bound_explicit_prefill_chunk"
            if config.prefill_chunk_tokens is not None
            else "source_bound_native_n_batch_prompt_fill"
        )
    elif profile == "generic":
        policy = authored.policy
        preemption_enabled = authored.preemption_enabled
        prefill_chunk_tokens = authored.prefill_chunk_tokens
        source = "authored_research_policy"
    else:
        policy = authored.policy
        preemption_enabled = config.preemption_enabled
        prefill_chunk_tokens = (
            config.prefill_chunk_tokens
            if config.prefill_chunk_tokens is not None
            else authored.prefill_chunk_tokens
        )
        source = "explicit_experimental_policy"
    resolved = replace(
        authored,
        max_num_seqs=config.parallel,
        max_num_batched_tokens=config.batch,
        max_num_ubatch_tokens=config.ubatch,
        mode="continuous" if config.cont_batching else "static",
        mixed_phase_batching=mixed_batching["status"] == "enabled",
        policy=policy,
        preemption_enabled=preemption_enabled,
        prefill_chunk_tokens=prefill_chunk_tokens,
        phase_candidate_order=slot_order["phase_candidate_order"],
    )
    return resolved, {
        "schema": "llama.cpp.effective-scheduler-policy/v1",
        "profile": profile,
        "source": source,
        "logical_batch_tokens": config.batch,
        "physical_ubatch_tokens": config.ubatch,
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "policy": policy,
        "preemption_enabled": preemption_enabled,
        "continuous_batching": config.cont_batching,
        "mixed_phase_batching": resolved.mixed_phase_batching,
        "phase_candidate_order": resolved.phase_candidate_order,
        "evidence_status": {
            "implemented": True,
            "source_derived": profile == "llama_cpp",
            "native_trace_validated": False,
            "timing_validated": False,
        },
    }



def llama_cpp_kv_layer_mapping(scenario: ScenarioConfig, config: LlamaCppRuntimeConfig) -> Mapping[str, Any]:
    """Return authoritative llama.cpp layer -> KV owner mapping."""
    view = model_graph_execution_view(scenario.model.graph, schema_version=scenario.model.schema_version)
    layers = tuple(item.layer for item in view.layer_instances if not item.layer.is_linear_attention)
    components = scenario.hardware.component_map()
    host = next((c.component_id for c in scenario.hardware.components if c.normalized_kind in {"host_memory", "dram", "ddr", "ddr_memory"} and c.is_active_memory and c.is_writable), None)
    if not config.offload_kqv:
        if host is None: raise ValueError("llama.cpp --no-kv-offload requires writable host memory")
        owner = {layer.layer_id: host for layer in layers}
    else:
        plan = build_parallel_plan(scenario, view)
        gpu_ranks = [r for r in plan.ranks if r.memory_component_id and r.memory_component_id in components]
        if not gpu_ranks:
            fallback = scenario.placement.kv_policy.cache_component
            if not fallback: raise ValueError("llama.cpp KV placement requires rank memory or cache component")
            owner = {layer.layer_id: fallback for layer in layers}
        elif config.split_mode == "layer":
            owner = {
                layer.layer_id: plan.ranks_for_layer(layer)[0].memory_component_id
                for layer in layers
            }
        else:
            main = next(
                (
                    r
                    for r in gpu_ranks
                    if r.pp_rank == 0 and r.tp_rank == config.main_gpu
                ),
                None,
            ) or gpu_ranks[min(config.main_gpu, len(gpu_ranks) - 1)]
            owner = {layer.layer_id: main.memory_component_id for layer in layers}
        # llama.cpp's ``-ngl N`` keeps the earliest repeating blocks on host
        # and places the last N loadable layers on GPU.  The output layer is
        # handled by the existing final-norm binding and is not a KV layer.
        if config.gpu_layers >= 0 and host is not None:
            gpu_start = max(0, len(layers) - int(config.gpu_layers))
            for index, layer in enumerate(layers):
                if index < gpu_start:
                    owner[layer.layer_id] = host
    for layer_id, component_id in owner.items():
        component = components.get(component_id)
        if component is None or not component.is_active_memory or not component.is_writable: raise ValueError(f"KV layer {layer_id} owner {component_id} must be writable active memory")
    plan = build_parallel_plan(scenario, view)
    ranks = {
        layer.layer_id: (
            [r.rank for r in plan.ranks_for_layer(layer)]
            if owner[layer.layer_id] != host
            else []
        )
        for layer in layers
    }
    return {"kv_layer_components": owner, "kv_layer_ranks": ranks, "split_mode": config.split_mode, "kv_unified": config.kv_unified, "offload_kqv": config.offload_kqv}

def llama_final_norm_static_binding(scenario: ScenarioConfig, config: LlamaCppRuntimeConfig) -> Mapping[str, Any]:
    """Bind the norm's own tensor; output-layer placement is only a candidate.

    Locked source: llama-model.cpp dev_output; llama-graph.cpp build_norm;
    ggml-backend.cpp weight preference and adjacent-backend expansion. Runtime
    buffer overrides/dispatch are not claimed as observed by this static rule.
    """
    metadata = dict(scenario.model.metadata)
    nested = scenario.model.graph.attributes.get("metadata", {})
    if isinstance(nested, Mapping):
        metadata.update(nested)
    raw = metadata.get("gguf_output_norm_binding", metadata.get("final_norm_weight_binding"))
    if not isinstance(raw, Mapping):
        return {"schema": "llama.cpp.final-norm-static/v1", "status": "weight_binding_missing",
                "native_dispatch_proven": False, "accuracy_validated": False}
    shape = raw.get("shape")
    if not isinstance(shape, (tuple, list)) or len(shape) != 1 or type(shape[0]) is not int or shape[0] <= 0:
        raise ValueError("output_norm.weight must have its own positive one-dimensional shape")
    view = model_graph_execution_view(scenario.model.graph, schema_version=scenario.model.schema_version)
    norm = next((op for op in view.operators if op.operator_id == "final_norm"), None)
    output = next((t for t in view.tensors if norm is not None and t.tensor_id in norm.output_tensor_ids), None)
    if output is None or not output.shape or shape[0] != output.shape[-1]:
        raise ValueError("output_norm.weight width differs from the model's final norm width")
    dtype = str(raw.get("type", "")).upper()
    bits = {"F32": 32, "F16": 16, "BF16": 16}.get(dtype)
    if bits is None or type(raw.get("n_bytes")) is not int or raw["n_bytes"] != shape[0] * bits // 8:
        raise ValueError("output_norm.weight requires exact scalar dtype and physical byte size")
    # A weight vector is not evidence about a custom graph's edges. Reuse
    # the existing opt-in, fixed-source ordinary-completion call-chain binding.
    # It covers the reviewed llama/qwen2/qwen35 build_norm(RMS, weight, NULL).
    from .final_layer_output_selection import model_declaration, resolve_declaration
    declaration = model_declaration(scenario.model)
    graph_policy = resolve_declaration(declaration, scenario.model.architecture,
                                      mtp_present=scenario.workload.mtp is not None)
    graph_pattern = ({"status": "fixed_source_call_chain", "declaration": dict(declaration),
                      "graph_architecture": scenario.model.architecture,
                      "call": "build_norm(cur, output_norm, NULL, LLM_NORM_RMS, -1)",
                      "source_rules": ("src/llama-graph.cpp:build_norm",
                                       "src/models/{llama,qwen2,qwen35}.cpp:output norm")}
                     if graph_policy is not None else {"status": "unknown_custom_graph"})
    return {"schema": "llama.cpp.final-norm-static/v1", "status": "source_conditional",
            # Only the norm's own scalar-storage fields belong here. Generic
            # artifact metadata walkers also visit runtime metadata, so copying
            # a tensor-directory block_size would poison unrelated GEMMs.
            "weight": {**{k: raw[k] for k in ("name", "type", "n_bytes", "offset") if k in raw},
                       "shape": tuple(shape), "bits": bits},
            "graph_pattern_binding": graph_pattern,
            "epsilon": metadata.get("gguf_norm_epsilon", metadata.get("final_norm_epsilon")),
            # This locked revision counts output in the tail: i_gpu_start is
            # n_layer_all + 1 - n_gpu_layers and dev_output uses il=n_layer_all.
            "output_device_candidate": "cpu" if config.gpu_layers == 0 else "rank_gpu",
            "output_device_rule": "locked_tail_including_output_layer",
            "model_source_sha256": "94ede4e7ac8119c5a4d2fad97e3432008ec7d30b42ab37395db6e2a8047d1984",
            "weight_buffer_selection": "native_output_candidate_list_then_per_tensor_compatibility",
            "scope": "RMS_NORM followed by its own one-dimensional weight MUL",
            "runtime_fingerprint": config.fingerprint,
            "source_rules": ("llama-graph.cpp:build_norm", "llama-model.cpp:dev_output",
                             "ggml-backend.cpp:backend_id_from_cur/pass2",
                             "ggml-cuda.cu:get_op_batch_size/device_supports_op"),
            "capacity_accounting": "included_in_existing_model_artifact_no_extra_allocation",
            "native_dispatch_proven": False, "accuracy_validated": False}


def apply_llama_runtime_config(
    scenario: ScenarioConfig,
    config: LlamaCppRuntimeConfig,
    *,
    materialize_placement: bool = True,
    recurrent_batching_contract: Mapping[str, Any] | None = None,
    slot_order_contract: Mapping[str, Any] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
) -> ScenarioConfig:
    """Return ``scenario`` with llama.cpp semantics lowered into typed fields."""
    if not isinstance(scenario, ScenarioConfig):
        raise TypeError("scenario must be a ScenarioConfig")
    if not isinstance(config, LlamaCppRuntimeConfig):
        raise TypeError("config must be a LlamaCppRuntimeConfig")
    identity = normalize_llama_runtime_identity(
        runtime_identity
        if runtime_identity is not None
        else scenario.workload.metadata.get("llama_cpp_runtime_identity")
    )
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
    scheduler, effective_policy = _resolve_llama_scheduler_policy(
        scenario, config, slot_order, mixed_batching
    )
    capabilities = {
        "runtime_config": {
            "implemented": True,
            "source_derived": identity["status"] == "bound",
            "native_trace_validated": bool(identity.get("native_trace_validated")),
            "timing_validated": bool(identity.get("timing_validated")),
        },
        "scheduler_policy": dict(effective_policy["evidence_status"]),
        "slot_order": {
            "implemented": True,
            "source_derived": slot_order.get("status") == "enabled",
            "native_trace_validated": False,
            "timing_validated": False,
        },
        "physical_batch_lowering": {
            "implemented": True,
            "source_derived": mixed_batching.get("status") == "enabled",
            "native_trace_validated": False,
            "timing_validated": False,
        },
    }
    workload = replace(
        scenario.workload,
        scheduler=scheduler,
        metadata={
            **scenario.workload.metadata,
            **hybrid_capabilities,
            **slot_metadata,
            "llama_cpp_runtime": config.to_dict(),
            "llama_cpp_runtime_identity": identity,
            "llama_cpp_final_norm_static": llama_final_norm_static_binding(scenario, config),
            "llama_cpp_mixed_phase_batching": mixed_batching,
            "llama_cpp_effective_scheduler_policy": effective_policy,
            "llama_cpp_capabilities": capabilities,
            "context_limit_semantics": "per_slot_runtime_limit",
            "llama_cpp_kv_capacity_contract": config.kv_capacity_contract(),
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
        kv_unified=config.kv_unified,
        # A runtime adapter invocation opts into native static layer
        # placement by default.  ``fixed`` and the explicit experimental pool
        # remain user-selected escape hatches.
        layout_mode=(
            "llama_static_layer"
            if kv.layout_mode not in {"fixed", "paged_pool"}
            else kv.layout_mode
        ),
    )
    placement_metadata = dict(scenario.placement.metadata)
    native_kv = llama_cpp_kv_layer_mapping(scenario, config)
    placement_metadata["llama_cpp_kv_layer_components"] = dict(native_kv["kv_layer_components"])
    placement_metadata["llama_cpp_kv_layer_ranks"] = dict(native_kv["kv_layer_ranks"])
    placement_metadata["llama_cpp_kv_contract"] = native_kv
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
    evidence["llama_cpp_runtime_identity"] = identity
    control["evidence"] = evidence
    placement_metadata["control_plane"] = control
    placement_metadata["llama_cpp_runtime"] = config.to_dict()
    placement_metadata["llama_cpp_runtime_fingerprint"] = config.fingerprint
    placement_metadata["llama_cpp_runtime_identity"] = identity
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
        ev.update({
            "llama_cpp_runtime": config.to_dict(),
            "llama_cpp_runtime_fingerprint": config.fingerprint,
            "llama_cpp_runtime_identity": identity,
        })
        cp["evidence"] = ev
        lowered = replace(lowered, placement=replace(lowered.placement, metadata={**metadata, "control_plane": cp}))
    return lowered


__all__ = ["apply_llama_runtime_config", "llama_final_norm_static_binding", "llama_cpp_kv_layer_mapping"]
