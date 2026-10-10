"""Real GGUF layer lowering must keep F32 intermediates separate from F16 KV."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from heterollm_sim.config import scenario_from_dict
from heterollm_sim import planner as p
from heterollm_sim.event_kernel import UnifiedEventKernel


ROOT = Path(__file__).resolve().parents[1]


def case(slug):
    return scenario_from_dict(json.loads((ROOT / "docs/frontend_native_validation_2026-10-07" /
        ("scenario_" + slug + "_512_128.json")).read_text(encoding="utf-8")))


def layer_tasks(scenario, tokens, prior=0, request_id=None):
    layer = next(item for item in p._execution_layers(scenario) if item.sequence_mixer == "full_attention")
    with p._compilation_scope(scenario):
        request = scenario.workload.requests[0]
        builder = p._TaskBuilder(replace(request, request_id=request_id) if request_id else request)
        p._compile_parallel_layer_body(builder, scenario, p._parallel_plan(scenario),
            p._topology_router(scenario), layer, token_batch=tokens, context_tokens=prior + tokens,
            kv_read_tokens=prior, kv_append_tokens=tokens, kv_materialized_tokens=tokens,
            linear_state_runtime=None, phase="decode" if tokens == 1 else "prefill", dependencies=())
    return layer, p._promote_physical_allocation_extents(builder.tasks)


def accesses(task, operation):
    return [row for row in task.metadata.get("memory_accesses", ()) if row["operation"] == operation]


def test_prefill_and_decode_set_rows_indices_have_separate_invocation_extents():
    scenario = case("qwen3_0_6b_f16")
    request = replace(scenario.workload.requests[-1], prompt_tokens=4,
                      output_tokens=3, arrival_ns=0)
    scenario = replace(scenario, workload=replace(scenario.workload, requests=(request,),
                                                 prompt_tokens=4, output_tokens=3))
    schedule = p.compile_scenario(scenario)
    indices = [row for task in schedule.tasks for row in accesses(task, "read")
               if row["buffer_id"].endswith("_set_rows:indices")]
    assert indices
    extents = {}
    for row in indices:
        assert row["buffer_id"].startswith("@invocation:")
        key = row["buffer_id"], row["allocation_generation"]
        extents.setdefault(key, row["allocation_size_bytes"])
        assert extents[key] == row["allocation_size_bytes"]
    k_extents = {row["allocation_size_bytes"] for row in indices
                 if row["buffer_id"].endswith(".k_set_rows:indices")}
    assert k_extents == {8, 32}


def test_set_rows_indices_are_reclaimed_after_closed_graph_use():
    scenario = case("qwen3_0_6b_f16")
    _, tasks = layer_tasks(scenario, 4)
    stores = tuple(replace(task, dependencies=()) for task in tasks
                   if task.metadata.get("event_kind") == "kv_native_set_rows"
                   and task.metadata.get("phase") != "kernel_launch")
    indices = [row for task in stores for row in accesses(task, "read")
               if row["buffer_id"].endswith("_set_rows:indices")]
    assert indices and all(row["allocation_generation"] > 0 for row in indices)
    kernel = UnifiedEventKernel.from_closed_graph(stores, capture_physical_details=False)
    while kernel.has_active_tasks:
        assert kernel.step() is not None
    kernel._reclaim_physical_allocations(float("inf"))
    for row in indices:
        assert kernel.physical_runtime.allocators[row["physical_owner"]].get_allocation(
            row["buffer_id"], row["allocation_generation"]) is None
    for task in stores:
        for row in accesses(task, "write"):
            assert row["allocation_generation"] == 0
            assert kernel.physical_runtime.allocators[row["physical_owner"]].get_allocation(
                row["buffer_id"], 0) is not None


def additional_family_case(preset_id):
    from heterollm_sim.gguf_model_catalog import inventory_to_gguf
    from heterollm_sim.gguf_parity import build_model_from_gguf
    record = json.loads((ROOT / "src/heterollm_sim/model_preset_data/gguf" /
                         (preset_id + ".json")).read_text(encoding="utf-8"))
    model = build_model_from_gguf(inventory_to_gguf(record["inventory"]))
    return replace(case("qwen3_0_6b_f16"), model=model)


def source_bound_case(scenario):
    from heterollm_sim.cuda_graph_lifecycle import SOURCE_REVISION
    # Isolate lowering: complete producer-file admission is covered by the
    # CUDA serving tests. These tasks are compiled without running a cohort.
    return replace(scenario, workload=replace(scenario.workload, metadata={
        **scenario.workload.metadata, "cuda_graph_structural_program": {
            "schema": "heterollm.cuda-graph-source-program/v1",
            "contract": {"source_revision": SOURCE_REVISION},
        }}))


def source_tail_tasks(scenario, tokens, prior=0, *, select=True):
    """Compile the actual last layer and output head, without executing GPU work."""
    with p._compilation_scope(scenario):
        plan, router = p._parallel_plan(scenario), p._topology_router(scenario)
        layer = p._execution_layers(scenario)[-1]
        builder = p._TaskBuilder(scenario.workload.requests[-1])
        builder._linear_state_owner_request_ids = (scenario.workload.requests[-1].request_id,)
        selection = p._final_output_selection(scenario, plan, tokens, (tokens - 1,)) if select else None
        phase = "decode" if tokens == 1 else "prefill"
        prepared, indices = p._prepare_output_selection_inputs(builder, scenario, router, plan,
            selection, phase, ())
        end = p._compile_parallel_layer_body(builder, scenario, plan, router, layer,
            token_batch=tokens, context_tokens=prior + tokens, kv_read_tokens=prior,
            kv_append_tokens=tokens, kv_materialized_tokens=tokens, linear_state_runtime=None,
            phase=phase, dependencies=prepared, output_selection=selection,
            output_indices_dependency=indices)
        p._compile_parallel_lm_head(builder, scenario, plan, router, phase, (end,),
            token_batch=1, output_selection=selection, output_indices_dependency=indices)
    return selection, p._promote_physical_allocation_extents(builder.tasks)


@pytest.mark.parametrize("architecture,position", [
    ("qwen2", "before_last_ffn"), ("qwen3", "before_last_ffn"),
    ("llama", "before_last_ffn"), ("qwen3_5_hybrid_transformer", "after_final_norm"),
])
def test_current_source_output_selection_has_its_own_revision_and_architecture_rule(architecture, position):
    from heterollm_sim.final_layer_output_selection import source_program_policy, source_declaration, resolve_declaration
    from heterollm_sim.cuda_graph_lifecycle import SOURCE_REVISION
    program = source_bound_case(case("qwen3_0_6b_f16")).workload.metadata["cuda_graph_structural_program"]
    policy = source_program_policy(program, architecture, mtp_present=False)
    prefill = policy.select(token_rows=512, selected_indices=(511,))
    assert prefill.position == position
    assert prefill.ffn_rows == (1 if position == "before_last_ffn" else 512)
    assert prefill.logit_rows == 1
    assert prefill.audit_metadata()["backend_commit"] == SOURCE_REVISION
    decode = policy.select(token_rows=1, selected_indices=(0,))
    assert decode.ffn_rows == decode.final_norm_rows == decode.logit_rows == 1
    legacy = resolve_declaration(source_declaration(), architecture, mtp_present=False)
    assert legacy.select(token_rows=512, selected_indices=(511,)).backend_commit == source_declaration()["backend_commit"]
    assert legacy.backend_commit != SOURCE_REVISION


@pytest.mark.parametrize("tokens,prior", [(1, 512), (16, 0)])
@pytest.mark.parametrize("preset", [None, "qwen2_5-0_5b", "llama3_2-1b"])
def test_source_output_selection_changes_only_last_ffn_rows_and_keeps_logit_rows(tokens, prior, preset):
    from heterollm_sim.serde import to_primitive
    scenario = source_bound_case(additional_family_case(preset) if preset else case("qwen3_0_6b_f16"))
    original_model = to_primitive(scenario.model)
    selection, selected = source_tail_tasks(scenario, tokens, prior)
    _, unselected = source_tail_tasks(scenario, tokens, prior, select=False)
    assert selection.logit_rows == selection.ffn_rows == 1
    assert to_primitive(scenario.model) == original_model
    def matrices(tasks, projection):
        return [task for task in tasks if task.metadata.get("phase") == "gpu_gemm"
                and task.metadata.get("projection_id") == projection]
    for projection in ("attention.qkv", "attention.output"):
        assert [task.demands for task in matrices(selected, projection)] == [task.demands for task in matrices(unselected, projection)]
        assert all(task.metadata["gemm_m"] == tokens for task in matrices(selected, projection))
    for projection in ("mlp.up_gate", "mlp.down"):
        assert all(task.metadata["gemm_m"] == 1 for task in matrices(selected, projection))
        assert all(task.metadata["gemm_m"] == tokens for task in matrices(unselected, projection))
        if tokens == 1:
            assert [task.demands for task in matrices(selected, projection)] == [task.demands for task in matrices(unselected, projection)]
    assert all(task.metadata["gemm_m"] == 1 for task in matrices(selected, "lm_head"))
    assert [task.demands for task in matrices(selected, "lm_head")] == [task.demands for task in matrices(unselected, "lm_head")]
    gathers = [task for task in selected if task.metadata.get("event_kind") == "output_row_selection"
               and task.metadata.get("phase") != "kernel_launch"]
    assert len(gathers) == 2  # Native retains GET_ROWS even for the one-token graph.
    assert all(task.metadata["output_selection_tensor_geometry"]["input_rows"] == tokens
               and task.metadata["output_selection_tensor_geometry"]["output_rows"] == 1 for task in gathers)
    index_buffers = []
    for gather in gathers:
        data_read, index_read = accesses(gather, "read")
        width = gather.metadata["output_selection_tensor_geometry"]["width"]
        assert data_read["offset_bytes"] == 4 * width * (tokens - 1)
        assert data_read["byte_count"] == 4 * width
        assert index_read["byte_count"] == 4 and index_read["offset_bytes"] == 0
        assert data_read["buffer_id"] != index_read["buffer_id"]
        index_buffers.append(index_read["buffer_id"])
    assert len(set(index_buffers)) == 1
    _, first = layer_tasks(scenario, tokens, prior)
    assert all(task.metadata["gemm_m"] == tokens for task in matrices(first, "mlp.up_gate"))


@pytest.mark.parametrize("tokens,prior", [(1, 512), (16, 0)])
def test_source_qwen35_output_selection_preserves_final_ffn_and_norm_rows(tokens, prior):
    scenario = source_bound_case(case("qwen3_8_27b_mixed"))
    selection, tasks = source_tail_tasks(scenario, tokens, prior)
    assert selection.position == "after_final_norm"
    assert selection.ffn_rows == selection.final_norm_rows == tokens
    assert selection.logit_rows == 1
    matrices = [task for task in tasks if task.metadata.get("phase") == "gpu_gemm"]
    ffn = [task for task in matrices if task.metadata.get("projection_id", "").startswith("mlp.")]
    assert ffn and all(task.metadata["gemm_m"] == tokens for task in ffn)
    head, = [task for task in matrices if task.metadata.get("projection_id") == "lm_head"]
    assert head.metadata["gemm_m"] == 1
    norms = [task for task in tasks if task.metadata.get("event_kind", "").startswith("final_norm")
             and task.metadata.get("phase") != "kernel_launch"]
    assert norms and all(task.metadata["final_layer_output_selection"]["final_norm_rows"] == tokens
                         for task in norms)
    gather, = [task for task in tasks if task.metadata.get("event_kind") == "output_row_selection"
               and task.metadata.get("phase") != "kernel_launch"]
    assert gather.metadata["final_layer_output_selection"]["stage"] == "final_norm_rows"
    assert gather.metadata["output_selection_tensor_geometry"]["input_rows"] == tokens
    data_read, index_read = accesses(gather, "read")
    width = gather.metadata["output_selection_tensor_geometry"]["width"]
    assert data_read["offset_bytes"] == 4 * width * (tokens - 1)
    assert data_read["byte_count"] == 4 * width and index_read["byte_count"] == 4
    assert data_read["buffer_id"] == accesses(norms[-1], "write")[0]["buffer_id"]
    launch, = [task for task in tasks if task.metadata.get("event_kind") == "output_row_selection"
               and task.metadata.get("phase") == "kernel_launch"]
    assert norms[-1].task_id in launch.dependencies and launch.task_id in gather.dependencies


def test_source_output_selection_rejects_conflicting_or_unproved_contracts():
    from heterollm_sim.final_layer_output_selection import source_declaration, source_program_policy
    scenario = source_bound_case(case("qwen3_0_6b_f16"))
    program = scenario.workload.metadata["cuda_graph_structural_program"]
    with pytest.raises(ValueError, match="MTP"):
        source_program_policy(program, "qwen3", mtp_present=True)
    with pytest.raises(ValueError, match="supported exact graph architecture"):
        source_program_policy(program, "unproved", mtp_present=False)
    with pytest.raises(ValueError, match="pinned CUDA source program"):
        source_program_policy({**program, "contract": {"source_revision": "unproved"}},
                              "qwen3", mtp_present=False)
    scenario = replace(scenario, model=replace(scenario.model, metadata={
        **scenario.model.metadata, "llama_cpp_final_layer_output_selection": source_declaration()}))
    with p._compilation_scope(scenario), pytest.raises(ValueError, match="conflicting"):
        p._final_output_selection(scenario, p._parallel_plan(scenario), 16, (15,))


@pytest.mark.parametrize("preset", ["qwen3-1_7b", "qwen3-4b"])
@pytest.mark.parametrize("tokens,prior", [(1, 512), (16, 0)])
def test_source_qkv_splits_real_matrices_even_when_formats_match(preset, tokens, prior):
    layer, tasks = layer_tasks(source_bound_case(additional_family_case(preset)), tokens, prior)
    matrices = [task for task in tasks if task.metadata.get("phase") == "gpu_gemm"
                and task.metadata.get("projection_id") == "attention.qkv"]
    assert len(matrices) == 3
    assert [task.metadata["physical_projection_segment_index"] for task in matrices] == [0, 1, 2]
    assert all(len(task.metadata["projection_segments"]) == 1 for task in matrices)
    assert sum(task.metadata["cost_model"]["weight_bytes"] for task in matrices) == sum(
        row["n_bytes"] for row in layer.metadata["gguf_tensor_bindings"]
        if row["name"].endswith((".attn_q.weight", ".attn_k.weight", ".attn_v.weight")))
    writes = [accesses(task, "write")[0] for task in matrices]
    assert len({row["buffer_id"] for row in writes}) == len({row["address"] for row in writes}) == 3
    for part, index in (("q", 0), ("k", 1)):
        norm = next(task for task in tasks if task.metadata.get("event_kind") == f"attention_{part}_norm_apply"
                    and task.metadata.get("phase") != "kernel_launch")
        read = next(row for row in accesses(norm, "read") if row["buffer_id"] == writes[index]["buffer_id"])
        assert read["address"] == writes[index]["address"]


@pytest.mark.parametrize("tokens,prior", [(1, 512), (16, 0)])
def test_source_bound_residual_stays_before_norm_and_generic_fusion_is_unchanged(tokens, prior):
    ordinary = case("qwen3_0_6b_f16")
    _, baseline = layer_tasks(ordinary, tokens, prior)
    assert any(task.metadata.get("event_kind") == "fused_residual_norm" for task in baseline)
    _, tasks = layer_tasks(source_bound_case(ordinary), tokens, prior)
    assert not any(task.metadata.get("event_kind") == "fused_residual_norm" for task in tasks)
    bodies = {task.metadata.get("event_kind"): task for task in tasks
              if task.metadata.get("phase") != "kernel_launch"}
    kinds = ("attention_residual", "post_attention_norm_reduce", "post_attention_norm_apply")
    selected = [bodies[kind] for kind in kinds]
    by_id = {task.task_id: task for task in tasks}
    def ancestors(task):
        seen, pending = set(), list(task.dependencies)
        while pending:
            ident = pending.pop()
            if ident not in seen:
                seen.add(ident)
                pending.extend(by_id[ident].dependencies)
        return seen
    assert selected[0].task_id in ancestors(selected[1])
    assert selected[1].task_id in ancestors(selected[2])
    assert all(task.metadata["fusion_decision"] == "source_boundary_requires_residual_before_norm"
               for task in selected)
    assert all(any(d.bytes_moved > 0 for d in task.demands) for task in selected)


@pytest.mark.parametrize("tokens,prior", [(1, 512), (16, 0)])
def test_source_attention_context_copy_is_physical_and_connects_pv_to_output(tokens, prior):
    ordinary = case("qwen3_0_6b_f16")
    layer, tasks = layer_tasks(source_bound_case(ordinary), tokens, prior)
    copies = [task for task in tasks if task.metadata.get("event_kind") == "attention_context_contiguous"]
    launch = [task for task in copies if task.metadata.get("phase") == "kernel_launch"]
    body, = [task for task in copies if task.metadata.get("phase") != "kernel_launch"]
    expected = 4 * tokens * layer.attention_heads * layer.effective_attention_head_dim
    assert len(launch) == (1 if tokens > 1 else 0)
    assert body.metadata["source_execution"] == ("f32_scalar_kernel" if tokens > 1 else "cuda_memcpy_d2d")
    assert body.metadata["cost_model"]["operations"] == 0
    assert body.metadata["cost_model"]["read_bytes"] == body.metadata["cost_model"]["write_bytes"] == expected
    assert sum(d.bytes_moved for d in body.demands if d.resource_id == "gddr0.gddr_fabric") == 2 * expected
    copy_read, = accesses(body, "read")
    copy_write, = accesses(body, "write")
    assert copy_read["buffer_id"] != copy_write["buffer_id"]
    pv = next(task for task in tasks if task.metadata.get("phase") == "gpu_gemm"
              and task.metadata["op_name"].endswith("attention_pv"))
    output = next(task for task in tasks if task.metadata.get("phase") == "gpu_gemm"
                  and task.metadata.get("projection_id") == "attention.output")
    pv_write, = accesses(pv, "write")
    output_read = next(row for row in accesses(output, "read") if row["buffer_id"] == copy_write["buffer_id"])
    assert (pv_write["buffer_id"], pv_write["address"]) == (copy_read["buffer_id"], copy_read["address"])
    assert output_read["address"] == copy_write["address"]
    assert all(row["byte_count"] == expected for row in (copy_read, copy_write, pv_write, output_read))
    _, previous = layer_tasks(ordinary, tokens, prior)
    assert not any(task.metadata.get("event_kind") == "attention_context_contiguous" for task in previous)
    old_pv = next(task for task in previous if task.metadata.get("phase") == "gpu_gemm"
                  and task.metadata["op_name"].endswith("attention_pv"))
    assert pv.demands == old_pv.demands  # Add the missing copy; never scale away the original GEMM.


@pytest.mark.parametrize("tokens,prior", [(1, 512), (16, 0)])
def test_source_quantized_ffn_keeps_both_real_matrix_boundaries(tokens, prior):
    scenario = source_bound_case(additional_family_case("qwen3-4b"))
    layer, tasks = layer_tasks(scenario, tokens, prior)
    matrices = [task for task in tasks if task.metadata.get("phase") == "gpu_gemm"
                and task.metadata.get("projection_id") in {"mlp.gate", "mlp.up"}]
    assert len(matrices) == 2
    assert {task.metadata["projection_id"] for task in matrices} == {"mlp.gate", "mlp.up"}
    assert all(task.metadata["gemm_m"] == tokens and task.metadata["gemm_n"] == layer.intermediate_size
               for task in matrices)
    physical_bytes = sum(row["n_bytes"] for row in layer.metadata["gguf_tensor_bindings"]
                         if row["name"].endswith((".ffn_gate.weight", ".ffn_up.weight")))
    assert sum(task.metadata["cost_model"]["weight_bytes"] for task in matrices) == physical_bytes
    assert any(task.metadata.get("event_kind") == "mlp_activation" for task in tasks)


@pytest.mark.parametrize("tokens,prior", [(1, 512), (16, 0)])
def test_source_f16_ffn_retains_only_the_supported_single_row_fusion(tokens, prior):
    layer, tasks = layer_tasks(source_bound_case(case("qwen3_0_6b_f16")), tokens, prior)
    matrices = [task for task in tasks if task.metadata.get("phase") == "gpu_gemm"
                and task.metadata.get("projection_id") == "mlp.up_gate"]
    assert len(matrices) == (1 if tokens == 1 else 2)
    assert sum(task.metadata["cost_model"]["weight_bytes"] for task in matrices) == 4 * layer.hidden_size * layer.intermediate_size
    assert any(task.metadata.get("event_kind") == "mlp_activation" for task in tasks) == (tokens > 1)


def test_source_lowering_rejects_wrong_revision_instead_of_using_generic_fusion():
    scenario = source_bound_case(case("qwen3_0_6b_f16"))
    scenario.workload.metadata["cuda_graph_structural_program"]["contract"]["source_revision"] = "wrong"
    with pytest.raises(ValueError, match="pinned structural compilation contract"):
        layer_tasks(scenario, 1, 512)


@pytest.mark.parametrize("preset_id", ["qwen2_5-0_5b", "llama3_2-1b", "llama3_1-8b"])
@pytest.mark.parametrize("tokens,prior", [(1, 512), (16, 0)])
def test_source_rope_without_qk_norm_reads_real_projection_and_frequency_inputs(preset_id, tokens, prior):
    scenario = additional_family_case(preset_id)
    layer, tasks = layer_tasks(scenario, tokens, prior)
    body = [t for t in tasks if t.metadata.get("phase") != "kernel_launch"]
    assert not any(t.metadata.get("event_kind", "").endswith("_norm_apply")
                   and t.metadata.get("norm_groups") for t in body)
    for operand, segment in (("q", 0), ("k", 1)):
        rope = next(t for t in body if t.metadata.get("event_kind") == "rope"
                    and t.metadata.get("rope_operand") == operand)
        reads = accesses(rope, "read")
        projected = next(r for r in reads if f"qkv:segment:{segment}" in r["buffer_id"])
        producers = [r for t in body for r in accesses(t, "write")
                     if r["buffer_id"] == projected["buffer_id"]]
        assert producers and all(r["address"] == projected["address"] for r in producers)
        assert not any("_norm" in r["buffer_id"] for r in reads)
        assert rope.metadata["rope_position_input_bytes"] == tokens * 4
        if preset_id.startswith("llama"):
            binding = rope.metadata["rope_frequency_factor_binding"]
            factor_read = next(r for r in reads if r["buffer_id"] == binding["name"])
            assert factor_read["byte_count"] == binding["n_bytes"] == layer.effective_attention_head_dim * 2
            assert rope.metadata["rope_frequency_factor_divisions"] == rope.metadata["rope_rotated_elements"] // 2
        else:
            assert "rope_frequency_factor_binding" not in rope.metadata
            bias = next(t for t in body if t.metadata.get("event_kind") == f"attention_{operand}_bias")
            assert projected["buffer_id"] in {r["buffer_id"] for r in accesses(bias, "write")}
    if preset_id.startswith("qwen"):
        bias_v = next(t for t in body if t.metadata.get("event_kind") == "attention_v_bias")
        store_v = next(t for t in body if t.metadata.get("event_kind") == "kv_native_set_rows"
                       and any("qkv:segment:2" in r["buffer_id"] for r in accesses(t, "read")))
        assert {r["buffer_id"] for r in accesses(bias_v, "write")} <= {r["buffer_id"] for r in accesses(store_v, "read")}


def test_native_llama_rope_rejects_missing_or_wrong_frequency_inventory():
    scenario = additional_family_case("llama3_2-1b")
    layer = p._execution_layers(scenario)[0]
    metadata = scenario.model.graph.attributes["metadata"]
    bindings = metadata.pop("gguf_rope_frequency_bindings")
    with pytest.raises(ValueError, match="frequency factor inventory"):
        p._source_rope_frequency_binding(scenario, layer, layer.effective_attention_head_dim)
    metadata["gguf_rope_frequency_bindings"] = bindings
    bindings[0]["n_bytes"] += 4
    with pytest.raises(ValueError, match="one F32 value"):
        p._source_rope_frequency_binding(scenario, layer, layer.effective_attention_head_dim)


def test_two_pass_softmax_reads_revisit_score_without_enlarging_its_allocation():
    scenario = additional_family_case("qwen3-14b")
    layer, tasks = layer_tasks(scenario, 512)
    qk = next(t for t in tasks if t.metadata.get("phase") == "gpu_gemm"
              and t.metadata["op_name"].endswith("attention_qk"))
    softmax = next(t for t in tasks if t.metadata.get("event_kind") == "softmax_normalize"
                   and t.metadata.get("phase") != "kernel_launch")
    extent = 4 * 512 * 512 * layer.attention_heads
    writes, reads = accesses(qk, "write"), accesses(softmax, "read")
    assert extent == 41943040
    assert len(writes) == 1 and writes[0]["allocation_size_bytes"] == extent
    assert softmax.metadata["input_buffer_read_passes"] == 2
    assert softmax.metadata["cost_model"]["read_bytes"] == 2 * extent
    backing_reads = softmax.metadata["cost_model"]["cache"]["logical_backing_read_bytes"]
    assert extent < backing_reads < 2 * extent
    assert [row["byte_count"] for row in reads] == [extent, backing_reads - extent]
    assert all(row["buffer_id"] == writes[0]["buffer_id"]
               and row["address"] == writes[0]["address"]
               and row["offset_bytes"] == 0
               and row["allocation_size_bytes"] == extent for row in reads)

    # The ordinary path must still reject a contiguous out-of-bounds read;
    # a known extent does not itself grant permission to wrap an access.
    metadata = {key: value for key, value in softmax.metadata.items()
                if key not in {"physical_memory_config", "memory_access", "memory_accesses",
                               "input_buffer_read_passes"}}
    with p._compilation_scope(scenario), pytest.raises(ValueError, match="access exceeds allocation"):
        p._attach_gddr_physical_task(replace(softmax, metadata=metadata), scenario)

    metadata["input_buffer_read_passes"] = 3
    with p._compilation_scope(scenario), pytest.raises(ValueError, match="exact declared pass count"):
        p._attach_gddr_physical_task(replace(softmax, metadata=metadata), scenario)


@pytest.mark.parametrize("slug", ["qwen3_0_6b_f16", "qwen3_8_27b_mixed"])
@pytest.mark.parametrize("tokens,prior", [(1, 512), (16, 0)])
def test_real_layer_f32_projections_norm_rope_cache_and_consumers(slug, tokens, prior):
    scenario = case(slug)
    layer, tasks = layer_tasks(scenario, tokens, prior)
    gemms = [t for t in tasks if t.metadata.get("phase") == "gpu_gemm"]
    qkv = [t for t in gemms if t.metadata.get("projection_id") == "attention.qkv"]
    assert len(qkv) == 3
    assert sum(t.metadata["cost_model"]["weight_bytes"] for t in qkv) == sum(
        row["n_bytes"] for row in layer.metadata["gguf_tensor_bindings"]
        if row["name"].endswith((".attn_q.weight", ".attn_k.weight", ".attn_v.weight")))
    for task in qkv:
        cost = task.metadata["cost_model"]
        assert cost["output_bytes"] == 4 * tokens * task.metadata["gemm_n"]
        assert task.metadata["modeled_kv_write_bytes"] == task.metadata["kv_materialized_bytes"] == 0
        assert sum(row["byte_count"] for row in accesses(task, "write")) == cost["output_bytes"]
    if slug == "qwen3_0_6b_f16":
        source = [t for t in gemms if t.metadata.get("mmvf_source_work")]
        assert len(source) == (6 if tokens == 1 else 0)
        if tokens == 1:
            assert all(t in source for t in qkv)
    qkv_ids = [accesses(t, "write")[0]["buffer_id"] for t in qkv]
    assert len(set(qkv_ids)) == 3
    for operand, index in (("q", 0), ("k", 1)):
        norm = next(t for t in tasks if t.metadata.get("event_kind") == "attention_" + operand + "_norm_apply"
                    and t.metadata.get("phase") != "kernel_launch")
        assert qkv_ids[index] in {r["buffer_id"] for r in accesses(norm, "read")}
        qkv_write = accesses(qkv[index], "write")[0]
        norm_read = next(r for r in accesses(norm, "read") if r["buffer_id"] == qkv_ids[index])
        assert qkv_write["address"] == norm_read["address"]
        rope = next(t for t in tasks if t.metadata.get("event_kind") == "rope"
                    and t.metadata.get("rope_operand") == operand and t.metadata.get("phase") != "kernel_launch")
        assert accesses(norm, "write")[0]["buffer_id"] in {r["buffer_id"] for r in accesses(rope, "read")}
        norm_write = accesses(norm, "write")[0]
        assert norm_write["address"] == next(r["address"] for r in accesses(rope, "read")
                                             if r["buffer_id"] == norm_write["buffer_id"])
    stores = [t for t in tasks if t.metadata.get("event_kind") == "kv_native_set_rows"
              and t.metadata.get("phase") != "kernel_launch"]
    assert len(stores) == 2
    width = layer.effective_kv_heads * layer.effective_attention_head_dim
    for part, task in zip(("k", "v"), stores):
        assert sum(row["byte_count"] for row in accesses(task, "write")) == 2 * tokens * width
        contract = task.metadata["native_kv_work"]
        assert contract["conversion_execution"] == "inside_set_rows_no_extra_conversion_launch"
        assert contract["conversion_instructions_priced"] is True
        assert contract["conversion_operations"] == tokens * width
        assert task.metadata["cost_model"]["operations"] == tokens * width
        cache_ids = {r["buffer_id"] for r in accesses(task, "write")}
        consumer = next(t for t in gemms if t.metadata["op_name"].endswith("attention_qk" if part == "k" else "attention_pv"))
        reads = [r for r in accesses(consumer, "read") if r["buffer_id"] in cache_ids]
        assert sum(r["byte_count"] for r in reads) == 2 * width * (768 if prior else 256)
        assert all(r["allocation_generation"] == 0 for r in reads + accesses(task, "write"))
        assert consumer.metadata["logical_context_tokens"] == prior + tokens
    append = [t for t in tasks if t.metadata.get("event_kind") == "kv_append"]
    assert len(append) == 1 and not append[0].demands
    assert append[0].metadata["resource_accounting"] == "native_cache_write_kernels"
    qk = next(t for t in gemms if t.metadata["op_name"].endswith("attention_qk"))
    by_id = {t.task_id: t for t in tasks}
    pending = list(qk.dependencies)
    ancestors = set()
    while pending:
        task_id = pending.pop()
        if task_id in ancestors:
            continue
        ancestors.add(task_id)
        pending.extend(by_id[task_id].dependencies if task_id in by_id else ())
    assert append[0].task_id in ancestors


def test_source_cache_views_are_bounded_and_persistent_between_steps():
    scenario = case("qwen3_0_6b_f16")
    layer = p._execution_layers(scenario)[0]
    plan = p._parallel_plan(scenario)
    router = p._topology_router(scenario)
    first = p._source_f32_kv_contract(scenario, router, plan, plan.ranks[0], layer, 512, 512, 512, 512)
    second = p._source_f32_kv_contract(scenario, router, plan, plan.ranks[0], layer, 1, 513, 1, 1)
    assert first["k_cache_id"] == second["k_cache_id"]
    assert first["v_cache_id"] == second["v_cache_id"]
    assert (first["attention_read_tokens"], second["attention_read_tokens"]) == (512, 768)
    for part in ("k", "v"):
        for contract in (first, second):
            for write in (True, False):
                for access in p._source_f32_kv_ranges(contract, part, write=write):
                    assert access["offset_bytes"] + access["size_bytes"] <= access["buffer_size_bytes"]


def test_serial_native_cache_requests_have_distinct_owners_and_reject_merged_sequences():
    scenario = case("qwen3_0_6b_f16")
    initial = scenario.workload.requests[0]
    requests = tuple(replace(initial, request_id=name) for name in ("startup", "warmup", "measured"))
    scenario = replace(scenario, workload=replace(scenario.workload, requests=requests))
    layer, plan, router = p._execution_layers(scenario)[0], p._parallel_plan(scenario), p._topology_router(scenario)
    contracts = [p._source_f32_kv_contract(scenario, router, plan, plan.ranks[0], layer,
        2, 2, 2, 2, owner_request_ids=(request.request_id,)) for request in requests]
    assert [contract["request_id"] for contract in contracts] == [request.request_id for request in requests]
    assert len({contract["k_cache_id"] for contract in contracts}) == 3
    assert len({contract["v_cache_id"] for contract in contracts}) == 3
    with pytest.raises(ValueError, match="single-GPU"):
        p._source_f32_kv_contract(scenario, router, plan, plan.ranks[0], layer,
            2, 2, 2, 2, owner_request_ids=("startup", "warmup"))
    with pytest.raises(ValueError, match="single-GPU"):
        p._source_f32_kv_contract(scenario, router, plan, plan.ranks[0], layer, 2, 2, 2, 2)


@pytest.mark.parametrize("cache_generation", [0, 1])
def test_source_cache_lifetime_in_indexed_kernel_graphs(cache_generation, monkeypatch):
    original_ranges = p._source_f32_kv_ranges
    def cache_ranges(contract, part, *, write=False):
        return [{**row, "allocation_generation": cache_generation}
                for row in original_ranges(contract, part, write=write)]
    monkeypatch.setattr(p, "_source_f32_kv_ranges", cache_ranges)
    scenario = case("qwen3_0_6b_f16")
    kernel = UnifiedEventKernel(resource_capacities=p._scenario_resource_capacities(scenario),
        resource_owners=p._scenario_resource_owners(scenario), capture_physical_details=False)
    addresses = {}
    for index, prior in enumerate((512, 513)):
        _, tasks = layer_tasks(scenario, 1, prior, request_id="cohort-" + str(index))
        stores = [t for t in tasks if t.metadata.get("event_kind") == "kv_native_set_rows"
                  and t.metadata.get("phase") != "kernel_launch"]
        for task in stores:
            for row in accesses(task, "write"):
                assert row["allocation_generation"] == cache_generation
        # Match the closed-cohort lifetime index used by serving. Dynamic
        # add_tasks alone intentionally cannot infer a buffer's last user.
        kernel.add_tasks(tasks)
        kernel._index_physical_allocation_uses(tasks)
        while kernel.has_active_tasks:
            assert kernel.step() is not None
        kernel._reclaim_physical_allocations(float("inf"))
        for task in stores:
            for row in accesses(task, "write")[:1]:
                owner = row["physical_owner"]
                allocation = kernel.physical_runtime.allocators[owner].get_allocation(row["buffer_id"], cache_generation)
                if cache_generation:
                    assert allocation is None, "transient generation must be reclaimed at the last cohort user"
                    continue
                assert allocation is not None
                if row["buffer_id"] in addresses:
                    assert allocation.base_address == addresses[row["buffer_id"]]
                addresses[row["buffer_id"]] = allocation.base_address
    kernel.release_request_physical_allocations(scenario.workload.requests[0].request_id)
    for task in stores:
        row = accesses(task, "write")[0]
        assert kernel.physical_runtime.allocators[row["physical_owner"]].get_allocation(row["buffer_id"], 0) is None
