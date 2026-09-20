"""Output-index placement follows actual GET_ROWS; synthetic graphs only."""
from dataclasses import replace
from graphlib import TopologicalSorter
from unittest.mock import patch

import pytest

from heterollm_sim import planner
from heterollm_sim.ir import RequestSpec
from heterollm_sim.llama_scenario import apply_llama_runtime_config
from tests.model_helpers import execution_layers, model_from_layer_specs
from tests.test_final_layer_output_selection_planner import _cohort
from tests.test_gpu_consumer_frontend import ancestors
from tests.test_llama_final_norm_placement import scenario
from tests.test_nonflash_kv_view import KEY as NONFLASH_KEY, scenario as nonflash_scenario
from tests.test_physical_projection_invocations import (
    _attention_execution_descriptors, _descriptors, _full_layer, _linear_layer,
)


ARCHITECTURES = ("llama_decoder", "qwen3_5_hybrid_transformer")
PLACEMENTS = ((0, False), (0, True), (1, True), (3, True))
ROWS = (1, 2, 4, 31, 32, 33, 64)


def case_for(architecture, ngl=0, offload=True):
    case = scenario(ngl=ngl, offload=offload)
    old = case.model
    model = model_from_layer_specs(old.name, execution_layers(old),
        vocabulary_size=old.vocabulary_size, max_sequence_length=old.max_sequence_length,
        embedding_weight_bytes=old.embedding_weight_bytes, metadata=old.metadata,
        architecture=architecture)
    return replace(case, model=model)


def assert_binding(schedule, logit_rows):
    tasks = schedule.tasks
    preparations = [t for t in tasks if t.metadata.get("event_kind") == "output_row_indices"]
    gathers = [t for t in tasks if t.metadata.get("event_kind") == "output_row_selection"]
    uploads = [t for t in tasks if t.metadata.get("event_kind") == "output_row_index_transfer"]
    assert len(preparations) == 1
    gpu_targets = {t.metadata["target_component"] for t in gathers
                   if t.metadata["target_component"].startswith("gpu")}
    assert len(uploads) == len(gpu_targets)
    assert {t.metadata["final_layer_output_selection"]["destination_component"] for t in uploads} == gpu_targets
    for upload in uploads:
        proof = upload.metadata["final_layer_output_selection"]
        assert proof["index_bytes"] == 4 * logit_rows
        assert proof["placement_source"] == "actual_output_row_selection_consumer"
        assert "physical_memory_traffic_status" not in proof
        assert "logical_read_bytes" not in proof
        assert preparations[0].task_id in ancestors(tasks, upload.task_id)
    for gather in gathers:
        target = gather.metadata["target_component"]
        required = next((t.task_id for t in uploads if
            t.metadata["final_layer_output_selection"]["destination_component"] == target), preparations[0].task_id)
        assert required in ancestors(tasks, gather.task_id)
    # ScheduleIR and segment capture support forward dependencies. Check the
    # actual DAG, not an incidental list order or the task-id sequence.
    ids = {task.task_id for task in tasks}
    assert len(ids) == len(tasks)
    for task in tasks:
        assert set(task.dependencies) <= ids
    assert len(tuple(TopologicalSorter({t.task_id: t.dependencies for t in tasks}).static_order())) == len(tasks)
    result = planner.execute_cost_schedule(schedule)
    assert len(result.execution_records) == len(tasks)
    return preparations, gathers, uploads


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("ngl,offload", PLACEMENTS)
@pytest.mark.parametrize("rows", ROWS)
def test_actual_consumer_controls_index_placement(architecture, ngl, offload, rows):
    case = case_for(architecture, ngl, offload)
    logits = rows if rows <= 4 else 1
    cohort = _cohort(rows, logits, context=64 if rows <= 4 else 0,
                     phase="decode" if rows <= 4 else "prefill")
    schedule = planner.compile_serving_cohort_schedule(case, cohort)
    _, gathers, uploads = assert_binding(schedule, logits)
    if architecture == "qwen3_5_hybrid_transformer" and ngl == 0:
        expected = "gpu0" if offload and rows >= 32 else "cpu0"
        assert {t.metadata["target_component"] for t in gathers} == {expected}
        assert len(uploads) == int(expected == "gpu0")


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_empty_indices_have_no_upload_or_gather(architecture):
    schedule = planner.compile_serving_cohort_schedule(case_for(architecture), _cohort(64, 0))
    preparations, gathers, uploads = assert_binding(schedule, 0)
    assert preparations[0].metadata["final_layer_output_selection"]["logical_write_bytes"] == 0
    assert not gathers and not uploads


def test_gpu_and_cpu_consumers_share_preparation_without_cross_device_wait():
    case = case_for("llama_decoder", 3, True)
    with planner._compilation_scope(case):
        plan = planner._parallel_plan(case)
        router = planner._topology_router(case)
        rank = plan.rank_at(0, 0, 0)
        layer = planner._execution_layers(case)[-1]
        builder = planner._TaskBuilder(RequestSpec("mixed", 0, 4, 1))
        selection = planner._final_output_selection(case, plan, 4, (3,))
        prior, indices = planner._prepare_output_selection_inputs(builder, case, router, plan, selection, "prefill", ())
        gpu, _ = planner._add_output_row_selection(builder, case, router, plan, rank, layer, selection,
            "gpu_gather", prior, stage="attention_output_rows", source_component="gpu0", target_component="gpu0",
            indices_dependency=indices, input_tensor_id="gpu_input")
        cpu, _ = planner._add_output_row_selection(builder, case, router, plan, rank, layer, selection,
            "cpu_gather", prior, stage="residual_input_rows", source_component="cpu0", target_component="cpu0",
            indices_dependency=indices, input_tensor_id="cpu_input")
        values = dict(builder._rank_value_components)
        previous, dma = builder.previous, builder._last_coherent_dma_task
        planner._bind_output_index_upload(builder, case, router, plan, 0, indices)
    uploads = [t for t in builder.tasks if t.metadata.get("event_kind") == "output_row_index_transfer"]
    assert len(uploads) == 1
    upload = uploads[0]
    assert upload.task_id in ancestors(builder.tasks, gpu)
    assert upload.task_id not in ancestors(builder.tasks, cpu)
    assert indices in ancestors(builder.tasks, cpu)
    assert builder._rank_value_components == values
    assert (builder.previous, builder._last_coherent_dma_task) == (previous, dma)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("ngl", (0, 3))
def test_static_request_prefill_and_decode_keep_per_invocation_indices(architecture, ngl):
    case = case_for(architecture, ngl, True)
    request = RequestSpec("static", 0, 64, 3)
    case = replace(case, workload=replace(case.workload, requests=(request,)))
    with planner._compilation_scope(case):
        tasks = planner._compile_parallel_request(case, request)
    indices = [t for t in tasks if t.metadata.get("event_kind") == "output_row_indices"]
    assert len(indices) == 3
    uploads = [t for t in tasks if t.metadata.get("event_kind") == "output_row_index_transfer"]
    assert len(uploads) == len({t.metadata["final_layer_output_selection"]["index_tensor_id"] for t in uploads})
    if architecture == "qwen3_5_hybrid_transformer" and ngl == 0:
        assert len(uploads) == 1  # M64 prefill offloads; subsequent M1 consumers remain CPU.


def serial_cache_case(ngl):
    """Real linear state plus source-described unfused QK scaling; 128-wide norm."""
    case = case_for("qwen3_5_hybrid_transformer", ngl, True)
    old = case.model
    layers = (
        replace(_linear_layer(_descriptors()), layer_id="layer0"),
        replace(_full_layer(_attention_execution_descriptors()), layer_id="layer1"),
    )
    weight = {**old.metadata["final_norm_weight_binding"], "shape": [128], "n_bytes": 512}
    model = model_from_layer_specs(old.name, layers, vocabulary_size=old.vocabulary_size,
        max_sequence_length=old.max_sequence_length, embedding_weight_bytes=64 * 128 * 4,
        metadata={**old.metadata, "final_norm_weight_binding": weight}, architecture=old.architecture)
    # Keep the native non-Flash cache's source-qualified physical-width contract,
    # not just its model name. This is the same bounded contract fixture used by
    # test_nonflash_kv_view, with the real runtime configuration matched here.
    runtime = replace(case.llama_cpp_config, flash_attn=False, kv_unified=True,
                      kv_type_k="f16", kv_type_v="f16")
    case = apply_llama_runtime_config(replace(case, model=model), runtime)
    view = nonflash_scenario(parallel=runtime.parallel, slot=runtime.context,
                            architecture=old.architecture).workload.metadata[NONFLASH_KEY]
    return replace(case, workload=replace(case.workload, metadata={**case.workload.metadata,
        NONFLASH_KEY: view, "supports_equal_length_stateful_ubatches": False}))


@pytest.mark.parametrize("ngl", (0, 3))
def test_norm_binding_invocation_cache_hit_is_exact(ngl):
    case = serial_cache_case(ngl)
    binding = case.workload.metadata["llama_cpp_final_norm_static"]
    assert binding["weight"]["shape"] == (128,) and binding["weight"]["type"] == "F32"
    assert binding["weight"]["n_bytes"] == 512
    assert binding["output_device_candidate"] == ("cpu" if ngl == 0 else "rank_gpu")
    cold = _cohort(1, 1, name="cold", context=9, phase="decode")
    warm = _cohort(1, 1, name="warm", context=9, phase="decode")
    group, = planner._serving_invocation_groups(case, cold)
    assert group.batching_semantics == "serial_stateful_position"
    assert group.nonflash_kv_view["applied"] is True
    assert group.nonflash_kv_view["physical_k_tokens"] == 256
    expected = planner.compile_serving_cohort_schedule(case, warm)
    context = planner.CompilationContext(case, eager_full_attention_segments=False,
                                         compiled_serving_invocation_segments=True)
    with patch.object(planner, "_compile_parallel_iteration", wraps=planner._compile_parallel_iteration) as body:
        with planner._compilation_scope(case, context):
            planner.compile_serving_cohort_schedule(case, cold)
            calls = body.call_count
            actual = planner.compile_serving_cohort_schedule(case, warm)
    assert calls > 0 and body.call_count == calls
    assert actual == expected
    _, gathers, uploads = assert_binding(actual, 1)
    target = "cpu0" if ngl == 0 else "gpu0"
    assert {task.metadata["target_component"] for task in gathers} == {target}
    norms = [task for task in actual.tasks if "final_norm_work" in task.metadata
             and task.metadata.get("phase") not in {"kernel_launch", "cpu_dispatch"}]
    assert norms and {task.metadata["target_component"] for task in norms} == {target}
    assert len(uploads) == int(ngl != 0)


@pytest.mark.parametrize("ngl", (0, 3))
def test_stateless_fixture_is_not_admitted_to_serial_invocation_cache(ngl):
    case = case_for("qwen3_5_hybrid_transformer", ngl, True)
    case = replace(case, workload=replace(case.workload, metadata={**case.workload.metadata,
        "supports_equal_length_stateful_ubatches": False}))
    cold = _cohort(1, 1, name="cold", context=9, phase="decode")
    warm = _cohort(1, 1, name="warm", context=9, phase="decode")
    group, = planner._serving_invocation_groups(case, cold)
    assert not any(layer.is_linear_attention for layer in execution_layers(case.model))
    assert group.batching_semantics == "stateless_scheduler_batch"
    expected = planner.compile_serving_cohort_schedule(case, warm)
    context = planner.CompilationContext(case, eager_full_attention_segments=False,
                                         compiled_serving_invocation_segments=True)
    decisions = []
    original_binding = planner._serving_invocation_segment_binding
    def record_binding(*args, **kwargs):
        result = original_binding(*args, **kwargs)
        decisions.append(result)
        return result
    with patch.object(planner, "_serving_invocation_segment_binding", side_effect=record_binding), \
         patch.object(planner, "_compile_parallel_iteration", wraps=planner._compile_parallel_iteration) as body:
        with planner._compilation_scope(case, context):
            planner.compile_serving_cohort_schedule(case, cold)
            calls = body.call_count
            actual = planner.compile_serving_cohort_schedule(case, warm)
    assert calls == 1 and body.call_count == 2 and decisions == [None, None]
    assert actual == expected
    assert_binding(actual, 1)


@pytest.mark.parametrize("ngl,rows", [(0, 1), (3, 64)])
def test_index_binding_online_compaction_preserves_dag_completion(ngl, rows):
    from heterollm_sim import serving
    case = case_for("qwen3_5_hybrid_transformer", ngl, True)
    plan = serving.compile_serving_plan(case)
    cohort = _cohort(rows, 1, context=64 if rows == 1 else 0,
                     phase="decode" if rows == 1 else "prefill")
    cohort = replace(cohort, items=tuple(replace(item, request_id=plan.requests[0].request_id) for item in cohort.items))
    with planner._compilation_scope(case):
        lowering = planner._lower_serving_cohort(case, cohort)
    detail = planner.execute_cost_schedule(lowering.schedule)
    production = planner.estimate_serving_cohort_cost(case, cohort)
    assert production["duration_ns"] == detail.makespan_ns
    stages, reason = serving._execution_stages_from_metadata(production["metadata"])
    assert reason is None and stages
    runtime = serving._OnlineRuntime(plan, lambda *_: serving.BatchCost(1.0))
    cost = serving.BatchCost(production["duration_ns"], metadata=production["metadata"])
    online = runtime._schedule_execution_stages(cohort, cost, stages, 0.0, detail.makespan_ns, 0.0)
    assert online[3] == pytest.approx(detail.makespan_ns, rel=0, abs=1e-7)
    assert online[-1] == pytest.approx(detail.makespan_ns, rel=0, abs=1e-7)


@pytest.mark.parametrize("ngl,rows", [(0, 32), (0, 64), (1, 1)])
def test_mixed_norm_backend_keeps_explicit_existing_stage_fallback(ngl, rows):
    """The adapter already rejects this mixed layer; do not claim online stage support."""
    from heterollm_sim import serving
    case = case_for("qwen3_5_hybrid_transformer", ngl, True)
    cohort = _cohort(rows, 1, context=64 if rows == 1 else 0,
                     phase="decode" if rows == 1 else "prefill")
    schedule = planner.compile_serving_cohort_schedule(case, cohort)
    assert_binding(schedule, 1)  # Closed, acyclic, fully executed actual task DAG.
    detail = planner.execute_cost_schedule(schedule)
    production = planner.estimate_serving_cohort_cost(case, cohort)
    assert production["duration_ns"] == detail.makespan_ns
    assert len(detail.execution_records) == len(schedule.tasks)
    metadata = production["metadata"]
    assert metadata["execution_stage_source"] == "serial_fallback"
    assert metadata["execution_stage_fallback_reason"] == "layer layer1 does not have exactly one compute component"
    stages, reason = serving._execution_stages_from_metadata(metadata)
    assert not stages and reason == "execution_stages is empty"
