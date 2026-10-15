"""Planner tasks consume stage metadata and the current placement decision."""
from dataclasses import replace

from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.control_plane_planner import plan_runtime_placement
from heterollm_sim.cost_models import (
    FusedAttentionWorkload, GemmWorkload, HBMProfile, estimate_gpu_gemm,
)
from heterollm_sim.ir import build_model_graph_from_layer_specs
from heterollm_sim.kernel_model import (
    KernelCapability, KernelModelProfile, estimate_kernel, summarize_kernel_predictions,
)
from heterollm_sim.planner import (
    _attach_planned_l2, _canonical_weight_tensor_id, _execution_layers,
)
from heterollm_sim.reference import build_reference_scenario
from test_current_kernel_callers import _gpu


def test_weight_aliases_have_one_decision_owner_after_planning_and_replanning():
    scenario = build_reference_scenario()
    graph = build_model_graph_from_layer_specs(scenario.model.name, _execution_layers(scenario),
        vocabulary_size=256, max_sequence_length=1024, tie_word_embeddings=True)
    scenario = replace(scenario, model=replace(scenario.model, graph=graph),
        workload=replace(scenario.workload, mtp=None),
        placement=replace(scenario.placement, metadata={
            **scenario.placement.metadata,
            "logical_weight_aliases": {"lm_head_weights": "obsolete-owner"},
        }))
    for _ in range(2):
        decision = plan_runtime_placement(scenario)
        assert decision.fully_placed
        scenario = decision.apply(scenario)
        assert "logical_weight_aliases" not in scenario.placement.metadata
        aliases = scenario.placement.metadata["control_plane"]["decision"]["logical_weight_aliases"]
        assert aliases["lm_head_weights"] == "embedding_weights"
        assert _canonical_weight_tensor_id(scenario, "lm_head_weights") == "embedding_weights"


def _phase_tasks(estimate):
    return tuple(TaskSpec(str(index), "request", phase.name, phase.category,
        demands=phase.demands, metadata={"phase": phase.name,
            "phase_metadata": dict(phase.metadata), "cost_model": dict(estimate.metadata)})
        for index, phase in enumerate(estimate.phases))


def test_stream_k_main_and_fixup_keep_their_own_stage_predictions():
    capability = KernelCapability("attention", ("fp16",), "fp16", "decode", "tensor",
        "analytical declaration", internal_dtype="fp16", operator="attention",
        attention_stream_k=True, attention_heads_per_tile=1, output_bits=32)
    gpu = _gpu(KernelModelProfile("gpu", "runtime", "arch", kernels=(capability,)))
    gpu = replace(gpu, tensor_core=replace(gpu.tensor_core, sm_count=8))
    estimate = estimate_kernel(gpu, HBMProfile(1000.), FusedAttentionWorkload(
        1, 1024, 64, score_heads=1, output_bits=32, execution_phase="decode"), attention=True)
    tasks = tuple(_attach_planned_l2(task) for task in _phase_tasks(estimate))
    assert {task.metadata["kernel_prediction"]["kernel_family"] for task in tasks} == {
        "attention", "flash_attn_stream_k_fixup_uniform"}
    aggregate = estimate.metadata["kernel_model"]
    auxiliary = TaskSpec("auxiliary", "request", "kv_materialization", TaskCategory.MEMORY,
        metadata={"phase": "kv_materialization", "phase_metadata": {},
                  "cost_model": dict(estimate.metadata)})
    assert "kernel_prediction" not in _attach_planned_l2(auxiliary).metadata
    summary = summarize_kernel_predictions((*tasks, auxiliary))
    assert summary["kernel_count"] == 2
    assert summary["fallback_count"] == 0
    by_family = {row["kernel_family"]: row for row in summary["groups"]}
    for task in tasks:
        audit = task.metadata["phase_metadata"]["kernel_model"]
        assert by_family[audit["kernel_family"]]["predicted_service_ns"] == audit["prediction"]["prediction_ns"]
        assert task.metadata["cost_model"]["kernel_model"] == aggregate


def test_generic_analytical_stage_predictions_remain_supported():
    estimate = estimate_gpu_gemm(_gpu(), HBMProfile(1000.),
        GemmWorkload(1, 16, 16, execution_phase="decode"))
    tasks = _phase_tasks(estimate)
    summary = summarize_kernel_predictions(tasks)
    assert summary["kernel_count"] == 0
    assert summary["fallback_count"] == sum(task.metadata["phase"] != "kernel_launch" for task in tasks)
    obsolete = tuple(replace(task, metadata={**task.metadata, "phase_metadata": {}}) for task in tasks)
    assert summarize_kernel_predictions(obsolete)["fallback_count"] == 0
