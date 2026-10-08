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
