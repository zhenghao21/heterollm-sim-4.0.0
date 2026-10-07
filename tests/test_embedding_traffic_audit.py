from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim import planner
from heterollm_sim.ir import LayerSpec, ModelSpec, build_model_graph_from_layer_specs
from heterollm_sim.llama_scenario import prepare_llama_scenario
from heterollm_sim.llama_tensor_storage import SCHEMA
from heterollm_sim.reference import build_llama_default_scenario
from heterollm_sim.reporting import _sum_batch_dram_traffic
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
from heterollm_sim.scalable_serving import execute_cost_schedule


def embedding_scenario(*, source_get_rows=False, f32_hidden=False):
    base = build_llama_default_scenario()
    graph = build_model_graph_from_layer_specs(
        "embedding-audit-fixture",
        (LayerSpec(layer_id="layer0", kind="dense", hidden_size=64,
                   intermediate_size=128, attention_heads=4, kv_heads=2,
                   dtype="fp16", weight_bytes=73728),),
        architecture="llama", vocabulary_size=32, max_sequence_length=4096,
        embedding_weight_bytes=4096, tie_word_embeddings=True,
        metadata={"gguf_sha256": "a" * 64,
                  "gguf_embedding_binding": {"shape": [64, 32], "type": "F16", "n_bytes": 4096}},
    )
    metadata = {"llama_cpp_f32_hidden_storage": f32_hidden}
    if source_get_rows:
        # A small declarative fixture for the already source-qualified contract;
        # these tests check lowering/physical billing, not source extraction.
        metadata["llama_cpp_tensor_storage_contract"] = {
            "schema": SCHEMA, "status": "source_derived",
            "embedding_index_bits": 32, "embedding_output_storage_bits": 32,
            "get_rows_access": "selected_packed_rows_per_index",
            "source_sha256": {"fixture-source": "b" * 64},
            "hidden_storage_evidence": {name: True for name in (
                "ggml_add_impl", "ggml_rms_norm_impl", "ggml_unary_impl",
                "ggml_ssm_conv", "ggml_gated_delta_net")},
            "native_latency_used": False, "accuracy_validated": False,
            "architectures": ["llama"],
            "supported_weight_types": {"cpu": ["F16"], "gpu": ["F16"]},
            "cpu_get_rows_tasks": 1,
        }
    request = replace(base.workload.requests[0], prompt_tokens=4, output_tokens=3)
    workload = replace(base.workload, requests=(request,), prompt_tokens=4,
                       output_tokens=3, metadata=metadata)
    return prepare_llama_scenario(replace(
        base, model=ModelSpec(name="embedding-audit-fixture", graph=graph),
        placement=replace(base.placement, model_name="embedding-audit-fixture",
                          parallel=replace(base.placement.parallel, layer_to_stage={})),
        workload=workload, llama_cpp_config=LlamaCppRuntimeConfig(gpu_layers=-1),
    ))


@pytest.mark.parametrize("source_get_rows,f32_hidden", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("token_rows", [1, 4])
def test_embedding_audit_matches_workload_and_physical_report(source_get_rows, f32_hidden, token_rows):
    scenario = embedding_scenario(source_get_rows=source_get_rows, f32_hidden=f32_hidden)
    detail = scenario.placement.metadata["control_plane"]["decision"]["weight_tensor_details"]
    assert detail["embedding_weights"]["runtime_copy_role"] == "input_embedding"
    with planner._compilation_scope(scenario):
        builder = planner._TaskBuilder(scenario.workload.requests[0])
        planner._compile_parallel_embedding(
            builder, scenario, planner._parallel_plan(scenario),
            planner._topology_router(scenario), "decode", (), token_batch=token_rows,
        )
    weight_bytes = token_rows * 128
    assert scenario.model.embedding_weight_bytes == 4096  # capacity stays whole-table
    index_bytes = token_rows * 4 if source_get_rows else 0
    output_bytes = token_rows * (256 if source_get_rows or f32_hidden else 128)
    marker = next(task for task in builder.tasks if task.metadata.get("event_kind") == "model_weight_access")
    memory = next(task for task in builder.tasks if task.name.endswith(".cpu_memory"))
    assert marker.metadata["weight_read_bytes"] == marker.metadata["bytes"] == weight_bytes
    assert memory.metadata["weight_read_bytes"] == weight_bytes
    assert memory.metadata["embedding_traffic_semantics"] == (
        "native_indexed_row_gather" if source_get_rows else "analytical_indexed_row_lookup")
    assert memory.metadata["cost_model"]["read_bytes"] == weight_bytes + index_bytes
    assert memory.metadata["cost_model"]["write_bytes"] == output_bytes
    assert sum(access["byte_count"] for access in memory.metadata["memory_accesses"]
               if access["operation"] == "read") == weight_bytes + index_bytes

    result = execute_cost_schedule(SimpleNamespace(
        tasks=tuple(builder.tasks), resource_capacities={}, resource_owners={}),
        retain_task_metadata=False)
    ledger = planner._summarize_dram_task_traffic(result.execution_records)
    batch = SimpleNamespace(cost=SimpleNamespace(metadata={"dram_traffic": ledger}))
    report = _sum_batch_dram_traffic(SimpleNamespace(serving=SimpleNamespace(batches=(batch,))))
    assert report["logical_read_bytes"] == weight_bytes + index_bytes
    assert report["logical_write_bytes"] == output_bytes
