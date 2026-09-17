"""Ordinary non-flash attention preserves the physical query-head score axis.

Tiny task graphs only: no native timing, device execution, or fitted constants.
The expected bytes follow [query_head, query_row, physical_KV_row], while
QK/PV FLOPs and the unique GQA cache storage retain their existing meaning.
"""
from collections import Counter
from dataclasses import replace
from unittest.mock import patch

import pytest

from heterollm_sim import planner
from heterollm_sim.ir import LayerSpec, ParallelSpec, RankMappingSpec
from tests.model_helpers import replace_model_layer_specs
from tests.test_final_layer_output_selection_planner import _cohort
from tests.test_nonflash_kv_view import scenario as nonflash_scenario
from tests.test_parallel import _two_gpu_nvlink_scenario
from tests.test_physical_projection_invocations import (
    _attention_execution_descriptors, _full_layer,
)


def _case(heads=4, kv_heads=2, *, hidden=None, architecture="qwen2", tp=1):
    case = nonflash_scenario(parallel=1, architecture=architecture)
    width = hidden or heads * 8
    layer = LayerSpec("score", "dense", hidden_size=width,
        intermediate_size=2 * width, attention_heads=heads,
        kv_heads=kv_heads, dtype="fp16")
    case = replace(case, model=replace_model_layer_specs(case.model, (layer,)),
        placement=replace(case.placement, kv_policy=replace(case.placement.kv_policy, dtype="fp16")),
        workload=replace(case.workload, prompt_tokens=4, output_tokens=1))
    if tp == 2:
        parallel = ParallelSpec(tp_degree=2, pp_degree=1, ep_degree=1,
            allow_padding=True, rank_mapping=(
                RankMappingSpec(0, "gpu0", 0, 0, 0, "hbm0"),
                RankMappingSpec(1, "gpu1", 1, 0, 0, "hbm0"),
            ))
        dual = _two_gpu_nvlink_scenario(parallel)
        # The source-qualified last-row selection fixture is single-rank.
        # This test exercises generic padded TP geometry without that unrelated
        # declaration, preserving its production single-rank guard.
        metadata = {key: value for key, value in case.model.metadata.items()
                    if key != "llama_cpp_final_layer_output_selection"}
        model = replace_model_layer_specs(replace(case.model, metadata=metadata),
                                          (layer,))
        case = replace(case, hardware=dual.hardware, model=model,
            placement=replace(case.placement, parallel=parallel))
    return case


def _capture(case, cohort=None, *, layer_only=False):
    with patch.object(planner, "_add_rank_gemm", wraps=planner._add_rank_gemm) as gemm, \
         patch.object(planner, "_add_rank_primitive", wraps=planner._add_rank_primitive) as primitive:
        if layer_only:
            # Isolate TP layer geometry from the single-rank native embedding
            # and output-row-selection contracts used by the serving fixture.
            builder = planner._TaskBuilder(planner.RequestSpec("tp_geometry", 0.0, 4, 1))
            plan = planner._parallel_plan(case)
            planner._compile_parallel_layer_body(builder, case, plan,
                planner._topology_router(case), planner._execution_layers(case)[0],
                token_batch=4, context_tokens=256, kv_read_tokens=256,
                kv_append_tokens=4, kv_materialized_tokens=4,
                linear_state_runtime=None, phase="tp_attention", dependencies=())
            schedule = builder
        else:
            schedule = (planner.compile_scenario(case) if cohort is None
                else planner.compile_serving_cohort_schedule(case, cohort))
    qk = [call.args[5] for call in gemm.call_args_list
          if call.args[5].name == "attention_qk_tp"]
    pv = [call.args[5] for call in gemm.call_args_list
          if call.args[5].name == "attention_pv_tp"]
    reduction = [call.args[6] for call in primitive.call_args_list
                 if call.args[6].name == "softmax_reduce_max_sum"]
    normalize = [call.args[6] for call in primitive.call_args_list
                 if call.args[6].name == "softmax_sub_exp_normalize"]
    payloads = {}
    task_ids = {task.task_id for task in schedule.tasks}
    for call in gemm.call_args_list:
        payloads.update({task_id: payload
            for task_id, payload in call.args[0]._task_segment_dynamic_payloads.items()
            if task_id in task_ids
            and isinstance(payload, planner._DynamicAttentionCostTaskReplayPayload)})
    return schedule, qk, pv, reduction, normalize, payloads


def _assert_score_geometry(captured, heads):
    schedule, qks, pvs, reductions, norms, payloads = captured
    assert qks and len(qks) == len(pvs) == len(reductions) == len(norms)
    for qk, pv, reduction, norm in zip(qks, pvs, reductions, norms):
        # Independent tensor geometry; no dependence on a descriptor being present.
        elements = heads * qk.m * qk.n
        assert pv.m == qk.m and pv.k == qk.n
        assert qk.output_bytes == pv.activation_bytes == 4 * elements
        assert reduction.input_elements == norm.elements == elements
        assert reduction.output_elements == heads * qk.m
        assert reduction.working_set_bytes == 4 * elements
        assert norm.working_set_bytes == 8 * elements
        # The folded GEMM already accounts for the query width in its reduction
        # or output dimension. Adding the score axis must not multiply FLOPs.
        assert qk.operations == 2 * qk.m * qk.k * qk.n
        assert pv.operations == 2 * pv.m * pv.k * pv.n
    assert payloads
    assert {payload.score_heads for payload in payloads.values()} == {heads}
    assert {payload.fused_workload.score_heads for payload in payloads.values()} == {heads}
    expected_groups = {heads * work.m for work in qks}
    assert all(task.metadata["softmax_reduction_groups"] in expected_groups
        for task in schedule.tasks if task.metadata.get("event_kind") == "softmax_reduce")


@pytest.mark.parametrize("architecture", ["qwen2", "llama_decoder"])
@pytest.mark.parametrize("heads,kv_heads", [(1, 1), (4, 4), (8, 2), (14, 2)])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_ordinary_mha_and_gqa_count_query_heads(architecture, heads, kv_heads, phase):
    case = _case(heads, kv_heads, architecture=architecture)
    assert planner._attention_execution_descriptor(planner._execution_layers(case)[0]) is None
    cohort = _cohort(4 if phase == "prefill" else 1, 0,
        context=0 if phase == "prefill" else 255, phase=phase)
    captured = _capture(case, cohort)
    _assert_score_geometry(captured, heads)
    assert all(work.n == 256 for work in captured[1])
    for qk, pv in zip(captured[1], captured[2]):
        assert qk.weight_bytes == pv.weight_bytes == 256 * kv_heads * 8 * 2
        assert qk.operations == pv.operations == 2 * qk.m * 256 * heads * 8


def test_static_and_serving_lowering_share_the_score_axis():
    case = _case(4, 2)
    _assert_score_geometry(_capture(case), 4)
    _assert_score_geometry(_capture(case, _cohort(4, 0)), 4)


def test_tp_head_padding_counts_physical_heads_not_fractional_heads():
    # Seven query heads cannot be evenly divided between two ranks. The existing
    # TP contract pads to four physical heads per rank; do not floor it to three.
    captured = _capture(_case(7, 1, tp=2), layer_only=True)
    _assert_score_geometry(captured, 4)
    assert len(captured[1]) == 2
    assert {payload.rank.tp_rank for payload in captured[-1].values()} == {0, 1}
    assert all(work.k == 28 for work in captured[1])  # folded width policy unchanged


def test_head_axis_does_not_multiply_launches_flops_or_unique_kv_bytes():
    # Same hidden width and same unique KV width (32), different query-head axes.
    small = _capture(_case(4, 2, hidden=64), _cohort(4, 0))
    large = _capture(_case(8, 4, hidden=64), _cohort(4, 0))
    for old, new in zip(small[1] + small[2], large[1] + large[2]):
        assert (new.m, new.k, new.n, new.operations, new.weight_bytes) == (
            old.m, old.k, old.n, old.operations, old.weight_bytes)
    assert large[1][0].output_bytes == 2 * small[1][0].output_bytes
    assert Counter(task.metadata.get("event_kind") for task in small[0].tasks) == Counter(
        task.metadata.get("event_kind") for task in large[0].tasks)
    assert sum(d.resource_id.endswith(".frontend") for t in small[0].tasks for d in t.demands) == sum(
        d.resource_id.endswith(".frontend") for t in large[0].tasks for d in t.demands)


def _payload_key(payload):
    return payload.layer.layer_id, payload.rank.rank, payload.role, payload.phase_index


@pytest.mark.parametrize("heads,kv_heads", [(1, 1), (4, 4), (14, 2)])
def test_dynamic_context_payload_refreshes_score_bytes_and_resources(heads, kv_heads):
    case = _case(heads, kv_heads)
    cold = _capture(case, _cohort(1, 0, name="cold", context=255, phase="decode"))
    warm = _capture(case, _cohort(1, 0, name="warm", context=256, phase="decode"))
    _assert_score_geometry(cold, heads)
    _assert_score_geometry(warm, heads)
    payloads = tuple(cold[-1].values())
    overrides = planner._task_segment_dynamic_task_overrides(payloads,
        planner._TaskSegmentDynamicReplayContext(case, 512, 512))
    assert overrides is not None and len(overrides) == len(payloads)
    warm_tasks = {task.task_id: task for task in warm[0].tasks}
    expected = {_payload_key(payload): warm_tasks[task_id]
                for task_id, payload in warm[-1].items()}
    for index, payload in enumerate(payloads):
        demands, metadata = overrides[index]
        target = expected[_payload_key(payload)]
        assert demands == target.demands
        assert metadata["cost_model"] == target.metadata["cost_model"]
        if payload.role in {"qk", "pv"}:
            assert metadata["kernel_query_geometry"] == target.metadata["kernel_query_geometry"]
    assert cold[1][0].output_bytes * 2 == warm[1][0].output_bytes
    # Ordinary unfused capture currently recompiles when its optional scale
    # stage is absent. Exercise that safe cache path too, without changing
    # admission rules merely to make a cache-hit assertion pass.
    context = planner.CompilationContext(case, eager_full_attention_segments=False,
                                        compiled_serving_invocation_segments=True)
    target = _cohort(1, 0, name="target", context=256, phase="decode")
    with planner._compilation_scope(case, context):
        planner.compile_serving_cohort_schedule(case,
            _cohort(1, 0, name="source", context=255, phase="decode"))
        actual = planner.compile_serving_cohort_schedule(case, target)
    assert actual == planner.compile_serving_cohort_schedule(case, target)


def test_specialized_descriptor_keeps_existing_geometry_and_true_dynamic_replay():
    case = nonflash_scenario(parallel=1, architecture="qwen3_5_hybrid_transformer")
    case = replace(case, model=replace_model_layer_specs(case.model,
        (_full_layer(_attention_execution_descriptors()),)))
    _assert_score_geometry(_capture(case, _cohort(1, 0, context=255, phase="decode")), 8)
    # A real stateful invocation needs the existing hybrid linear-attention
    # prefix. A stateless one-layer fixture is intentionally not admitted to
    # that cache; do not weaken the cache guard to manufacture a hit.
    first = planner._execution_layers(nonflash_scenario(
        parallel=1, architecture="qwen3_5_hybrid_transformer"))[0]
    first = replace(first, hidden_size=128, intermediate_size=256)
    case = replace(case, model=replace_model_layer_specs(case.model,
        (first, _full_layer(_attention_execution_descriptors()))))
    cold = _cohort(1, 1, name="cold", context=255, phase="decode")
    warm = _cohort(1, 1, name="warm", context=256, phase="decode")
    expected = planner.compile_serving_cohort_schedule(case, warm)
    context = planner.CompilationContext(case, eager_full_attention_segments=False,
                                        compiled_serving_invocation_segments=True)
    with patch.object(planner, "_compile_parallel_iteration",
                      wraps=planner._compile_parallel_iteration) as compile_body:
        with planner._compilation_scope(case, context):
            planner.compile_serving_cohort_schedule(case, cold)
            count = compile_body.call_count
            actual = planner.compile_serving_cohort_schedule(case, warm)
    assert compile_body.call_count == count
    assert actual == expected
