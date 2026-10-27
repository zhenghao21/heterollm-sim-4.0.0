"""Conditional same-shape operator reuse for approved Maverick LONG Prefill layouts.

This module is intentionally opt-in. It retains real chunk lowering and
resource/lane guards while reusing only representative layer-window terms.
"""
from copy import deepcopy
import json
import time

def operator_reuse_sample_prefill_chunks(runtime, cohort, stages, request_ready_ns, control, options, records):
    """Approximate chunk sampler: execute at most two representative windows.

    Every chunk is still lowered by the ordinary producer, preserving its
    logical KV geometry and address diagnostics. Only the physical layer
    window is reused. The first chunk is always sampled; a second sample is
    taken at the first distinct full-attention boundary. All other chunks use
    the nearest representative's terms, so this function is intentionally
    experimental and conditional, never a formal native continuation.
    """
    from heterollm_sim.native_prefill import sample_prefill
    from heterollm_sim.native_prefill_chunk_inputs import iter_prefill_chunk_inputs
    from heterollm_sim.native_prefill_chunk_adapter import build_prefill_chunk_terms
    from heterollm_sim.native_prefill_chunk_composition import compose_prefill_chunks
    from heterollm_sim.native_stationary import SAME_CLASS_ADDRESS_ASSUMPTION

    graphs, chunks, extra_edges, assumptions = [], [], [], []
    representatives, origin, frontier = {}, None, None
    started = __import__("time").perf_counter()

    def attention_key(prepared):
        geometry = prepared.get("kv_geometry") or ()
        return tuple((row["layer_id"], row.get("attention_chunk_size"),
                      row.get("historical_context_within_attention_block")) for row in geometry)

    def has_full_attention(prepared):
        return any(row.get("attention_chunk_size") == 0 for row in (prepared.get("kv_geometry") or ()))

    def remap_prefix_tasks(sampled, prepared, representative_stage_keys):
        """Map representative prefix timing by stable stage/order geometry.

        Chunk task IDs and stage IDs are freshly namespaced by lowering. We
        therefore pair sorted current stages with sorted representative stage
        groups, then let the adapter verify resource sets and lane equality.
        No address or event is fabricated.
        """
        prefix_ids = tuple(prepared["boundary"].get("before_first_layer", ()))
        specs = {task.task_id: task for task in prepared["specs"]}
        current_by_stage = {}
        current_stage_ids = {}
        for identity in prefix_ids:
            stage = specs[identity].metadata.get("execution_stage_id")
            key = prepared["stage_key_by_id"].get(stage, stage)
            current_by_stage.setdefault(key, []).append(identity)
            current_stage_ids[key] = stage
        current_stages = sorted(current_by_stage, key=repr)
        if not current_stages:
            raise RuntimeError("operator_reuse_prefix_task_count_changed")
        for run_name in ("calibration", "entry_at_calibration_origin", "entry_at_holdout_origin"):
            run = sampled["runs"].get(run_name)
            if run is None or "prefill_terms" not in run:
                raise RuntimeError("operator_reuse_prefix_terms_missing")
            source_by_stage = {}
            for row in run["prefill_terms"].get("prefix_tasks", ()):
                source_stage = row.get("stage_id")
                key = representative_stage_keys.get(source_stage)
                if key is None and isinstance(source_stage, str) and source_stage.startswith("native-stage:"):
                    try:
                        decoded = json.loads(source_stage[len("native-stage:"):])
                        key = tuple(decoded) if isinstance(decoded, list) else decoded
                    except (TypeError, ValueError, json.JSONDecodeError):
                        key = source_stage
                if key is None:
                    key = source_stage
                source_by_stage.setdefault(key, []).append(row)
            source_stages = sorted(source_by_stage, key=repr)
            if not source_stages and current_stages:
                # Some native stage captures intentionally omit prefix rows.
                # Keep the approximation explicit by making the current
                # prefix an unmeasured declared boundary; no task is invented.
                prepared["boundary"]["before_first_layer"] = []
                prepared["chunk_input_boundary"].setdefault("assumptions", []).append(
                    "operator_reuse_prefix_rows_unavailable_declared_boundary")
                for current_run in ("calibration", "entry_at_calibration_origin", "entry_at_holdout_origin"):
                    if current_run in sampled["runs"]:
                        sampled["runs"][current_run]["prefill_terms"]["prefix_tasks"] = []
                return sampled
            if source_stages != current_stages:
                if (len(source_stages) == len(current_stages)
                        and [len(source_by_stage[key]) for key in source_stages]
                        == [len(current_by_stage[key]) for key in current_stages]):
                    # IDs differ, but stable sorted stage/group order and
                    # task counts match. The adapter retains resource/lane
                    # guards after this explicit opt-in remap.
                    prepared["chunk_input_boundary"].setdefault("assumptions", []).append(
                        "operator_reuse_stage_id_remap_by_group_order")
                else:
                    raise RuntimeError("operator_reuse_prefix_stage_geometry_changed:" + repr({"current": current_stages, "source": source_stages}))
            mapped = []
            for current_stage, source_stage in zip(current_stages, source_stages):
                identities = sorted(current_by_stage[current_stage], key=str)
                rows = sorted(source_by_stage[source_stage],
                              key=lambda row: (row.get("start_ns", 0.), row.get("end_ns", 0.),
                                               str(row.get("node_id"))))
                if len(rows) != len(identities):
                    raise RuntimeError("operator_reuse_prefix_stage_geometry_changed")
                for identity, source in zip(identities, rows):
                    row = deepcopy(source)
                    row["node_id"], row["stage_id"] = identity, current_stage_ids[current_stage]
                    mapped.append(row)
            run["prefill_terms"]["prefix_tasks"] = mapped
        return sampled

    for prepared in iter_prefill_chunk_inputs(runtime, cohort, stages, request_ready_ns, control, options):
        control.raise_if_cancelled()
        if origin is None:
            origin = prepared["origin"]
        prepared["tp_symmetry"] = (runtime.plan.scenario, options.tp_rank_symmetry)
        prepared["chunk_separate_prefix"] = True
        key = attention_key(prepared)
        source = "reused_operator_terms"
        representative_index = None
        if not representatives:
            sampled = sample_prefill(prepared, control, options, records)
            representatives[key] = (prepared["chunk_index"], deepcopy(sampled), has_full_attention(prepared), deepcopy(prepared["stage_key_by_id"]))
            source, representative_index = "native_representative_chunk", prepared["chunk_index"]
        elif key in representatives:
            representative_index, sampled_template, _, representative_stage_keys = representatives[key]
            sampled = deepcopy(sampled_template)
        elif len(representatives) < 2 and has_full_attention(prepared):
            sampled = sample_prefill(prepared, control, options, records)
            representatives[key] = (prepared["chunk_index"], deepcopy(sampled), has_full_attention(prepared), deepcopy(prepared["stage_key_by_id"]))
            source, representative_index = "native_attention_representative_chunk", prepared["chunk_index"]
        else:
            # Prefer the attention representative for later full-attention
            # geometry, otherwise the first representative.
            preferred = next((item for item in representatives.values()
                              if item[2] == has_full_attention(prepared)), None)
            representative_index, sampled_template, _, representative_stage_keys = (preferred if preferred is not None
                                                       else next(iter(representatives.values())))
            sampled = deepcopy(sampled_template)
        if source == "reused_operator_terms":
            sampled = remap_prefix_tasks(sampled, prepared, representative_stage_keys)
        try:
            adapted = build_prefill_chunk_terms(prepared, sampled, prepared["chunk_index"], frontier=frontier)
        except ValueError as error:
            raise RuntimeError("operator_reuse_chunk_terms_guard: " + str(error)) from error
        frontier = adapted["frontier"]
        graphs.append(adapted["chunk"])
        extra_edges.extend(adapted["extra_edges"])
        assumptions.extend(adapted["assumptions"])
        assumptions.extend(prepared["chunk_input_boundary"]["assumptions"])
        chunks.append({"context_tokens": prepared["context_tokens"],
            "prompt_tokens": prepared["chunk_prompt_tokens"], "run_layers": sampled["layers"],
            "row_state_bound_basis": "representative_operator_physical_window_terms",
            "prefix_separate_from_class_samples": True, "input_boundary": prepared["chunk_input_boundary"],
            "kv_geometry": prepared["kv_geometry"], "wall_seconds": sampled["wall_seconds"],
            "layer_intervals": adapted["layer_intervals"], "sampled_class_parts": adapted["diagnostics"].get("sampled_class_parts"),
            "operator_reuse": {"source": source, "representative_chunk_index": representative_index,
                "attention_key": key, "address_state_diagnostics": {
                    "source_boundary_is_real": prepared["chunk_input_boundary"].get("source_boundary_is_real"),
                    "address_equivalence_proven": prepared["chunk_input_boundary"].get("address_equivalence_proven"),
                    "future_physical_state_advanced": prepared["chunk_input_boundary"].get("future_physical_state_advanced")},
                "production_results_eligible": False}})
        control.report("native_prefill_chunk_sample_complete", len(chunks), unit="sampled_chunks",
            message="算子复用实验已完成分段 lowering；仅代表块执行物理样本",
            metadata={"chunk_index": prepared["chunk_index"], "sample_source": source,
                      "representative_chunk_index": representative_index,
                      "full_cohort_committed": False})
    total = next(iter(runtime.states.values())).spec.prompt_tokens
    if sum(row["prompt_tokens"] for row in chunks) != total:
        raise RuntimeError("operator_reuse_complete_chunk_inputs_not_observed")
    result = compose_prefill_chunks(graphs, origin_ns=origin,
        target_relative_error=options.target_relative_error,
        assumptions=[SAME_CLASS_ADDRESS_ASSUMPTION,
            "operator_reuse_same_shape_terms_are_conditioned_on_representative_physical_window; "
            "future address/controller state is diagnostic only", *assumptions], extra_edges=extra_edges)
    result["error_budget"].update(prompt_tokens=total, chunks=chunks,
        sampled_chunk_count=len(chunks), native_chunk_count=0,
        conditional_accuracy=True, experimental=True, default_enabled=False,
        physical_full_cohorts_committed=0, physical_full_prompt_executed=False,
        sampled_layer_window_count=len(records), sample_wall_seconds=__import__("time").perf_counter() - started,
        algorithm="operator_reuse_same_shape_terms_v1",
        operator_reuse={"enabled": True, "representative_chunk_count": len(representatives),
            "reused_chunk_count": len(chunks) - len(representatives),
            "attention_boundary_sampled": len(representatives) > 1,
            "address_state_conditioned_diagnostic": True,
            "production_results_eligible": False})
    return result



__all__ = ["operator_reuse_sample_prefill_chunks"]
