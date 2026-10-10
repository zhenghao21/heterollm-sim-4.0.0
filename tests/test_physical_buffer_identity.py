"""Physical storage survives changes in logical projection and cohort shape."""
from dataclasses import replace

import pytest

from heterollm_sim import planner as p
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import UnifiedEventKernel
from test_gddr_runtime_integration import _gpu_gddr_scenario


def weight_task(scenario, projection, cohort, **metadata):
    task = TaskSpec(
        task_id=f"{cohort}.{projection}", request_id=f"cohort-{cohort}", name=projection,
        category=TaskCategory.MEMORY,
        demands=(ResourceDemand("hbm0.memory", 0, bytes_moved=64),),
        metadata={"target_component": "gpu0", "rank": 0,
                  "projection_id": projection, "op_name": projection,
                  "weight_tensor_id": "layer-000.mlp_weights",
                  "weight_buffer_id": "blk.0.ffn_gate.weight",
                  "cost_model": {"activation_bytes": 0, "weight_bytes": 64, "output_bytes": 0},
                  **metadata},
    )
    return p._attach_gddr_physical_task(task, scenario)


@pytest.mark.parametrize("identity", [
    {"weight_buffer_id": "blk.0.ffn_gate.weight"},
    {"weight_buffer_id": None, "weight_allocation_id": "gguf-gate-allocation"},
])
def test_same_physical_weight_is_not_allocated_again_for_a_fused_projection(identity):
    scenario = _gpu_gddr_scenario()
    tasks = [weight_task(scenario, projection, index, **identity)
             for index, projection in enumerate(("mlp.gate", "mlp.up_gate"))]
    rows = [task.metadata["memory_accesses"][0] for task in tasks]
    assert rows[0]["buffer_id"] == rows[1]["buffer_id"]
    assert all(row["allocation_generation"] == 0 for row in rows)
    kernel = UnifiedEventKernel.from_closed_graph(tasks, capture_physical_details=False)
    while kernel.has_active_tasks:
        kernel.step()
    allocations = kernel.physical_runtime.allocators["hbm0.memory"].allocations()
    assert len(allocations) == 1


def test_physical_identity_keeps_different_weights_ranks_and_logical_shards_separate():
    scenario = _gpu_gddr_scenario()
    rows = [weight_task(scenario, "mlp.gate", 0, **meta).metadata["memory_accesses"][0]
            for meta in ({}, {"weight_buffer_id": "blk.0.ffn_up.weight"}, {"rank": 1})]
    assert len({row["buffer_id"] for row in rows}) == 3
    logical = [weight_task(scenario, projection, 0, weight_buffer_id=None)
               for projection in ("mlp.gate", "mlp.up")]
    assert len({task.metadata["memory_accesses"][0]["buffer_id"] for task in logical}) == 2


def test_weight_gather_scratch_changes_generation_and_is_reclaimed():
    scenario = _gpu_gddr_scenario()
    kernel = UnifiedEventKernel(capture_physical_details=False)
    for cohort, size in enumerate((40, 10)):
        task = TaskSpec(
            task_id=f"gather-{cohort}", request_id=f"cohort-{cohort}",
            name="router.weight_gather", category=TaskCategory.MEMORY,
            demands=(ResourceDemand("hbm0.memory", 0, bytes_moved=size),),
            metadata={"target_component": "gpu0", "op_name": "router.weight_gather",
                      "buffer_accesses": [{"buffer_id": "read:operator=router.weight_gather:ordinal=0",
                          "operation": "read", "offset_bytes": 0, "size_bytes": size,
                          "buffer_size_bytes": size, "allocation_generation": 0}]},
        )
        task = p._attach_gddr_physical_task(task, scenario)
        access = task.metadata["memory_accesses"][0]
        assert access["allocation_generation"] == cohort + 1
        kernel.add_tasks((task,))
        kernel._index_physical_allocation_uses((task,))
        while kernel.has_active_tasks:
            kernel.step()
        kernel._reclaim_physical_allocations(float("inf"))
        assert not kernel.physical_runtime.allocators["hbm0.memory"].allocations()


def test_explicit_resident_identity_does_not_require_the_word_weight():
    scenario = _gpu_gddr_scenario()
    task = weight_task(scenario, "mlp.gate", 1, weight_buffer_id="matrix-A")
    row = task.metadata["memory_accesses"][0]
    explicit = replace(task, metadata={key: value for key, value in task.metadata.items()
                                      if key not in {"physical_memory_config", "physical_owner"}})
    rebound = p._attach_gddr_physical_task(explicit, scenario)
    assert rebound.metadata["memory_accesses"][0]["allocation_generation"] == row["allocation_generation"] == 0


def test_empty_allocation_id_does_not_mask_a_physical_buffer():
    scenario = _gpu_gddr_scenario()
    plain = weight_task(scenario, "mlp.gate", 0)
    empty = weight_task(scenario, "mlp.gate", 0, weight_allocation_id="")
    assert plain.metadata["memory_accesses"] == empty.metadata["memory_accesses"]


def test_real_gguf_f16_fused_decode_reuses_both_prefill_weight_allocations():
    from test_source_f32_attention_storage import case, layer_tasks, source_bound_case
    scenario = source_bound_case(case("qwen3_0_6b_f16"))
    weights = []
    for tokens, prior in ((16, 0), (1, 16)):
        _, tasks = layer_tasks(scenario, tokens, prior, request_id=f"cohort-{tokens}")
        rows = {}
        for task in tasks:
            if task.metadata.get("projection_id") != "mlp.up_gate":
                continue
            for row in task.metadata.get("memory_accesses", ()):
                if any(f"blk.0.ffn_{part}.weight" in row["buffer_id"] for part in ("gate", "up")):
                    rows[row["buffer_id"]] = (row["byte_count"], row["allocation_size_bytes"],
                                               row["allocation_generation"])
        assert len(rows) == 2
        assert sum(row[0] for row in rows.values()) == 12 * 1024 * 1024
        weights.append(rows)
    assert weights[0] == weights[1]
