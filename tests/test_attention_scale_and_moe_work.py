"""Mandatory scaled attention and weighted MoE work reaches physical costs."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim import planner
from heterollm_sim.ir import LayerSpec, ModelSpec, RankMappingSpec, build_model_graph_from_layer_specs
from heterollm_sim.llama_scenario import prepare_llama_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
from heterollm_sim.scalable_serving import execute_cost_schedule
from test_embedding_traffic_audit import embedding_scenario
from test_full_attention_projection_geometry import _scenario


def _moe_case(tp=1, ep=1, shared=False, cpu=False):
    base = embedding_scenario()
    layer = LayerSpec(layer_id="layer0", kind="moe", hidden_size=64,
                      intermediate_size=32, attention_heads=4, kv_heads=2,
                      dtype="fp16", weight_bytes=90112 + (6272 if shared else 0), num_experts=4, experts_per_token=2,
                      shared_expert_intermediate_size=16 if shared else 0, shared_expert_gate=shared,
                      metadata={"moe_routing": {"gating": "softmax", "normalize_selected_weights": True,
                                                 "weight_scale": 1.0}})
    graph = build_model_graph_from_layer_specs("moe-work", (layer,), architecture="qwen3",
                vocabulary_size=32, max_sequence_length=4096, embedding_weight_bytes=4096,
                tie_word_embeddings=True)
    result = prepare_llama_scenario(replace(base, model=ModelSpec(name="moe-work", graph=graph),
                         placement=replace(base.placement, model_name="moe-work",
                                           parallel=replace(base.placement.parallel, layer_to_stage={})),
                         llama_cpp_config=LlamaCppRuntimeConfig(gpu_layers=0 if cpu else -1)))
    result = replace(result, placement=replace(result.placement, parallel=replace(
        result.placement.parallel, tp_degree=tp, ep_degree=ep,
        rank_mapping=tuple(RankMappingSpec(rank=e*tp+t, component_id="gpu0", tp_rank=t,
                            pp_rank=0, ep_rank=e, memory_component_id="hbm0")
                           for e in range(ep) for t in range(tp)))))
    return result


def _compile_moe(scenario, tokens=4):
    with planner._compilation_scope(scenario):
        builder = planner._TaskBuilder(scenario.workload.requests[0])
        planner._compile_parallel_moe(builder, scenario, planner._parallel_plan(scenario),
            planner._topology_router(scenario), planner._execution_layers(scenario)[0],
            tokens, "decode.layer0", ())
    return builder


@pytest.mark.parametrize("tokens", (1, 4))
def test_unfused_ordinary_attention_charges_scale_once(tokens):
    scenario = _scenario("qwen3-0_6b")
    layer = planner._execution_layers(scenario)[0]
    with planner._compilation_scope(scenario):
        builder = planner._TaskBuilder(scenario.workload.requests[0])
        planner._compile_parallel_layer_body(builder, scenario, planner._parallel_plan(scenario),
             planner._topology_router(scenario), layer, token_batch=tokens, context_tokens=7,
             kv_read_tokens=6, kv_append_tokens=1, kv_materialized_tokens=7,
             linear_state_runtime=None, phase="decode", dependencies=())
    costs = [t for t in builder.tasks if t.metadata.get("event_kind") == "softmax_normalize"
             and "cost_model" in t.metadata]
    assert costs
    for task in costs:
        assert task.metadata["qk_scale"] == pytest.approx(1 / layer.effective_attention_head_dim**0.5)
        assert task.metadata["qk_scale_accounting"] == "fused_into_softmax"
        assert task.metadata["cost_model"]["operations"] == 3 * tokens * 7 * layer.attention_heads
    assert not any(t.metadata.get("event_kind") == "attention_qk_scale" for t in builder.tasks)


@pytest.mark.parametrize("tp,ep", ((1,1),(2,1),(1,2),(2,2)))
def test_moe_global_routing_weighting_and_distinct_weight_addresses(tp, ep):
    scenario = _moe_case(tp, ep)
    builder = _compile_moe(scenario)
    by_id = {t.task_id: t for t in builder.tasks}
    def ancestors(task):
        found, pending = set(), list(task.dependencies)
        while pending:
            key = pending.pop()
            if key not in found:
                found.add(key)
                pending.extend(by_id[key].dependencies)
        return found
    mats = [t for t in builder.tasks if t.metadata.get("phase") == "gpu_gemm"
            and "expert_index" in t.metadata]
    assert len(mats) == 4 * tp * 2
    weight_accesses = [next(a for a in t.metadata["memory_accesses"]
                           if a["operation"] == "read" and "expert_weights" in a["buffer_id"])
                       for t in mats]
    assert len({a["buffer_id"] for a in weight_accesses}) == len(weight_accesses)
    assert len({a["address"] for a in weight_accesses}) == len(weight_accesses)
    costs = [t for t in builder.tasks if "cost_model" in t.metadata and t.metadata.get("phase") != "kernel_launch"]
    for kind in ("router_weight_gather", "router_weight_sum", "router_weight_clamp",
                 "router_weight_normalize", "expert_route_weight", "expert_weighted_sum"):
        assert sum(t.metadata.get("event_kind") == kind for t in costs) == tp
    routers = [t for t in costs if t.metadata.get("event_kind") == "router_softmax_normalize"]
    for task in routers:
        assert task.metadata["cost_model"]["operations"] == 2 * 4 * 4
        assert any("router_logits_all_gather" in by_id[k].name for k in ancestors(task))
    for task in costs:
        if task.metadata.get("event_kind") == "expert_route_weight":
            assert task.metadata["cost_model"]["operations"] == 4 * 2 * 64
        if task.metadata.get("event_kind") == "expert_weighted_sum":
            assert task.metadata["cost_model"]["operations"] == 4 * 64 * (2 - 1)
    # Execute the complete little graph: this catches physical allocation,
    # dependency, and service-contract errors that counting nodes would miss.
    result = execute_cost_schedule(SimpleNamespace(tasks=tuple(builder.tasks),
              resource_capacities={}, resource_owners={}), retain_task_metadata=False)
    assert result.makespan_ns > 0


def test_fused_attention_uses_same_explicit_scale_once(monkeypatch):
    scenario = _scenario("qwen3-0_6b")
    layer = planner._execution_layers(scenario)[0]
    captured = []
    original = planner.FusedAttentionWorkload
    def capture(**kwargs):
        result = original(**kwargs)
        captured.append(result)
        return result
    monkeypatch.setattr(planner, "FusedAttentionWorkload", capture)
    with planner._compilation_scope(scenario):
        builder = planner._TaskBuilder(scenario.workload.requests[0])
        planner._compile_parallel_layer_body(builder, scenario, planner._parallel_plan(scenario),
             planner._topology_router(scenario), layer, token_batch=2, context_tokens=7,
             kv_read_tokens=6, kv_append_tokens=1, kv_materialized_tokens=7,
             linear_state_runtime=None, phase="decode", dependencies=())
    assert len(captured) == 1
    work = captured[0]
    assert work.qk_scale == pytest.approx(1 / layer.effective_attention_head_dim**0.5)
    assert work.qk_scale_operations == 2 * 7 * layer.attention_heads
    assert work.scalar_operations == work.softmax_scalar_operations + work.qk_scale_operations + work.kv_dequant_operations


def test_fused_scale_reaches_scalar_resource_demand():
    from heterollm_sim.cost_models import FusedAttentionWorkload, estimate_gpu_fused_attention
    scenario = _scenario("qwen3-0_6b")
    rank = planner._parallel_plan(scenario).ranks[0]
    gpu, memory = planner._gpu_profiles(scenario, rank.component_id, rank.memory_component_id)
    workload = FusedAttentionWorkload(batch_tokens=4, context_tokens=7, hidden_size=2048,
                                     score_heads=16, qk_scale=1 / 128**0.5)
    with_scale = estimate_gpu_fused_attention(gpu, memory, workload)
    without_scale = estimate_gpu_fused_attention(gpu, memory, replace(workload, qk_scale=None))
    def work(cost):
        return sum(d.work_units for phase in cost.phases for d in phase.demands
                   if d.resource_id == gpu.scalar_resource_id)
    assert work(with_scale) - work(without_scale) == 4 * 7 * 16


def test_qwen3_moe_refuses_missing_route_normalization_contract():
    scenario = _moe_case()
    layer = replace(planner._execution_layers(scenario)[0], metadata={})
    with planner._compilation_scope(scenario), pytest.raises(ValueError, match="explicit moe_routing"):
        planner._compile_parallel_moe(planner._TaskBuilder(scenario.workload.requests[0]), scenario,
             planner._parallel_plan(scenario), planner._topology_router(scenario), layer,
             1, "decode.layer0", ())


def test_ep_exchange_keeps_full_hidden_width_of_tp_partial_outputs(monkeypatch):
    scenario = _moe_case(tp=2, ep=2)
    original = planner._add_collective_tasks
    exchanged = []
    def capture(*args, **kwargs):
        if args[5] == "all_to_all":
            exchanged.append((args[4], args[7]))
        return original(*args, **kwargs)
    monkeypatch.setattr(planner, "_add_collective_tasks", capture)
    _compile_moe(scenario, tokens=4)
    assert len(exchanged) == 4  # dispatch + combine on each TP coordinate
    assert all(byte_count == 4 * 2 * 64 * 2 for _, byte_count in exchanged)


def test_shared_expert_has_distinct_weights_and_one_sigmoid_per_token():
    scenario = _moe_case(shared=True)
    builder = _compile_moe(scenario, tokens=4)
    matrices = [t for t in builder.tasks if t.metadata.get("phase") == "gpu_gemm"
                and t.metadata.get("ffn_path") == "shared"]
    assert len(matrices) == 3
    weights = [next(a for a in t.metadata["memory_accesses"]
                    if a["operation"] == "read" and "shared_expert" in a["buffer_id"])
               for t in matrices]
    assert len({a["buffer_id"] for a in weights}) == len({a["address"] for a in weights}) == 3
    gate, = [t for t in builder.tasks if t.metadata.get("ffn_op") == "shared_gate_apply"
             and "cost_model" in t.metadata and t.metadata.get("phase") != "kernel_launch"]
    assert gate.metadata["cost_model"]["operations"] == 4 * 64 + 3 * 4
    assert gate.metadata["cost_model"]["special_function_operations"] == 4
    assert sum(d.work_units for d in gate.demands if d.resource_id.endswith(".sfu")) == 4
    assert gate.metadata["cost_model"]["read_bytes"] == (4 * 64 + 4) * 2
    residual, = [t for t in builder.tasks if t.metadata.get("event_kind") == "moe_residual"
                 and "cost_model" in t.metadata and t.metadata.get("phase") != "kernel_launch"]
    assert residual.metadata["cost_model"]["operations"] == 2 * 4 * 64
    result = execute_cost_schedule(SimpleNamespace(tasks=tuple(builder.tasks),
              resource_capacities={}, resource_owners={}), retain_task_metadata=False)
    assert result.makespan_ns > 0


def test_new_moe_work_respects_cpu_layer_placement():
    scenario = _moe_case(cpu=True)
    builder = _compile_moe(scenario)
    names = {"router_weight_gather", "router_weight_sum", "router_weight_clamp",
             "router_weight_normalize", "expert_route_weight", "expert_weighted_sum"}
    operations = [t for t in builder.tasks if t.metadata.get("event_kind") in names
                  and "cost_model" in t.metadata]
    assert {t.metadata["event_kind"] for t in operations} == names
    assert all(not any(d.resource_id.startswith("gpu0.") for d in t.demands) for t in operations)
    assert all(any(d.resource_id.startswith("cpu0.") for d in t.demands) for t in operations)
