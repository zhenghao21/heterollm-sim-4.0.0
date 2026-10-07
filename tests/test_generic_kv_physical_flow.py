"""KV consumers must read current rows from the same independently owned K/V buffers."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim import planner as p
from heterollm_sim.ir import RequestSpec
from test_direct_backing_physical_ownership import prepared_case


def compile_layer(scenario, *, rows=1, history=512, phase="decode", builder=None):
    layer = p._execution_layers(scenario)[0]
    with p._compilation_scope(scenario):
        builder = builder or p._TaskBuilder(scenario.workload.requests[0])
        p._compile_parallel_layer_body(builder, scenario, p._parallel_plan(scenario), p._topology_router(scenario),
            layer, token_batch=rows, context_tokens=history + rows,
            kv_read_tokens=history, kv_append_tokens=rows, kv_materialized_tokens=rows,
            linear_state_runtime=None, phase=phase, dependencies=())
    return layer, builder.tasks


def rows_of(task, operation):
    return [row for row in task.metadata.get("memory_accesses", ()) if row["operation"] == operation]


@pytest.mark.parametrize("hardware", ("local", "b200_2hbf", "b200_3hbf"))
@pytest.mark.parametrize("model", ("qwen3-0_6b", "qwen2_5-0_5b"))
def test_current_kv_row_is_physically_written_then_read(hardware, model):
    scenario = prepared_case(hardware, model)
    layer, tasks = compile_layer(scenario)
    matrices = {task.name.split(".")[-2]: task for task in tasks if task.metadata.get("phase") == "gpu_gemm"}
    one_row = p._kv_tensor_bytes(scenario, layer, 1, 1)
    qkv = matrices["qkv"]
    assert qkv.metadata["modeled_kv_write_bytes"] == 0
    assert qkv.metadata["cost_model"]["output_bytes"] == (
        layer.attention_heads + 2 * layer.effective_kv_heads) * layer.effective_attention_head_dim * 2
    writes = [row for task in tasks if task.metadata.get("event_kind") == "kv_materialize"
              for row in rows_of(task, "write") if row.get("buffer_id", "").startswith("kv:")]
    assert len(writes) == 2
    assert {row["buffer_id"].rsplit(":", 1)[1] for row in writes} == {"k", "v"}
    assert all(row["offset_bytes"] == 512 * one_row and row["byte_count"] == one_row for row in writes)
    assert all(row["allocation_generation"] == 0 for row in writes)
    for role, name in (("k", "attention_qk"), ("v", "attention_pv")):
        task = matrices[name]
        assert task.metadata["cost_model"]["weight_bytes"] == 513 * one_row
        reads = [row for row in rows_of(task, "read") if row.get("buffer_id", "").endswith(":" + role)]
        assert sum(row["byte_count"] for row in reads) == 513 * one_row
        assert {row["buffer_id"] for row in reads} == {row["buffer_id"] for row in writes if row["buffer_id"].endswith(":" + role)}
        assert all(row["allocation_generation"] == 0 for row in reads)
    # Separate current-row stores are causal predecessors, not a disconnected side branch.
    by_id = {task.task_id: task for task in tasks}
    pending = list(matrices["attention_qk"].dependencies)
    ancestors = set(pending)
    while pending:
        pending.extend(dep for dep in by_id[pending.pop()].dependencies if dep not in ancestors and not ancestors.add(dep))
    assert all(task.task_id in ancestors for task in tasks if task.metadata.get("event_kind") == "kv_materialize")


def test_prefill_has_materialized_rows_even_with_no_historical_kv():
    scenario = prepared_case()
    layer, tasks = compile_layer(scenario, rows=4, history=0, phase="prefill")
    row_bytes = p._kv_tensor_bytes(scenario, layer, 1, 1)
    qk = next(task for task in tasks if task.name.endswith("attention_qk.gpu_gemm"))
    assert qk.metadata["cost_model"]["weight_bytes"] == 4 * row_bytes
    ranges = qk.metadata["rhs_buffer_accesses"]
    assert len(ranges) == 1 and ranges[0]["size_bytes"] == 4 * row_bytes


def test_request_identity_survives_cohorts_and_slot_reuse_does_not_alias():
    scenario = prepared_case()
    first = replace(scenario.workload.requests[0], request_id="first", prompt_tokens=8, output_tokens=4)
    second = replace(first, request_id="second")
    scenario = replace(scenario, workload=replace(scenario.workload, requests=(first, second)))
    layer = p._execution_layers(scenario)[0]
    with p._compilation_scope(scenario):
        plan = p._parallel_plan(scenario)
        def contract(request_id, history, cohort):
            builder = p._TaskBuilder(RequestSpec(cohort, 0, 1, 1))
            builder._attention_invocation_lanes = (SimpleNamespace(request_id=request_id,
                context_tokens=history + 1, kv_read_tokens=history, kv_materialized_tokens=1),)
            return p._generic_kv_contract(builder, scenario, plan, plan.ranks[0], layer,
                token_batch=1, context_tokens=history + 1, history_tokens=history,
                materialized_tokens=1)
        a, b, c = contract("first", 8, "cohort-1"), contract("first", 9, "cohort-2"), contract("second", 8, "cohort-3")
    assert a["owners"][0]["id"] == b["owners"][0]["id"] != c["owners"][0]["id"]
    assert p._generic_kv_accesses(a, "k", write=True)[0]["offset_bytes"] != p._generic_kv_accesses(b, "k", write=True)[0]["offset_bytes"]
    assert p._generic_kv_accesses(a, "k")[0]["buffer_id"] != p._generic_kv_accesses(a, "v")[0]["buffer_id"]
    assert p._generic_kv_persistence(a)["persistent_request_buffers"] == {"first": [a["owners"][0]["id"] + ":k", a["owners"][0]["id"] + ":v"]}


def test_declared_physical_scan_already_includes_current_row():
    scenario = prepared_case()
    layer = p._execution_layers(scenario)[0]
    with p._compilation_scope(scenario):
        plan = p._parallel_plan(scenario)
        builder = p._TaskBuilder(scenario.workload.requests[0])
        builder._attention_invocation_lanes = (SimpleNamespace(request_id=scenario.workload.requests[0].request_id,
            context_tokens=513, kv_read_tokens=512, kv_materialized_tokens=1),)
        contract = p._generic_kv_contract(builder, scenario, plan, plan.ranks[0], layer,
            token_batch=1, context_tokens=640, history_tokens=640, materialized_tokens=1, includes_current=True)
    assert contract["read_tokens"] == 640
    assert contract["owners"][0]["write_positions"] == (512,)


def test_artifact_unpack_reuses_declared_format_work_without_inventing_quantizer():
    scenario = prepared_case()
    scenario = replace(scenario, placement=replace(scenario.placement,
        kv_policy=replace(scenario.placement.kv_policy, dtype="q4_0")))
    layer = p._execution_layers(scenario)[0]
    workload = p._layer_gemm(layer, 1, 16, 8, name="attention_qk", dynamic_rhs=True)
    actual = p._generic_kv_gemm_dequant(scenario, layer, 1, 3, workload)
    _bits, spec = p._kv_dtype_bits(scenario, layer)
    assert actual.epilogue_operations == 3 * (p._physical_kv_width_for_rank(layer, 1) // spec.block_size) * spec.block_size * spec.dequant_operations_per_weight
    assert actual.epilogue_name == "dequant_Q4_0"

@pytest.mark.parametrize("hardware,kind", (("local", "host_memory"), ("b200_2hbf", "hbf")))
def test_staged_ddr_and_direct_hbf_use_the_same_cache_range_contract(hardware, kind):
    scenario = prepared_case(hardware)
    layer = p._execution_layers(scenario)[0]
    cache = next(component.component_id for component in scenario.hardware.components if component.kind == kind)
    metadata = dict(scenario.placement.metadata)
    metadata["llama_cpp_kv_layer_components"] = {layer.layer_id: cache}
    metadata["memory_tiers"] = {**metadata.get("memory_tiers", {}), "kv_layer_components": {layer.layer_id: cache}}
    scenario = replace(scenario, placement=replace(scenario.placement, metadata=metadata))
    layer, tasks = compile_layer(scenario, history=16)
    row_bytes = p._kv_tensor_bytes(scenario, layer, 1, 1)
    if kind == "host_memory":
        stores = [task for task in tasks if task.metadata.get("event_kind") == "kv_store"
                  and task.metadata.get("component_id") == cache]
        reads = [task for task in tasks if task.metadata.get("event_kind") == "kv_read"
                 and task.metadata.get("component_id") == cache]
        assert len(stores) == len(reads) == 2
        written = {row["buffer_id"] for task in stores for row in task.metadata.get("source_buffer_accesses", ())}
        assert written == {row["buffer_id"] for task in reads for row in task.metadata["source_buffer_accesses"]}
        assert all(task.metadata["source_buffer_accesses"][0]["offset_bytes"] == 16 * row_bytes for task in stores)
        assert all(task.metadata["source_buffer_accesses"][0]["size_bytes"] == 17 * row_bytes for task in reads)
        qk = next(task for task in tasks if task.name.endswith("attention_qk.gpu_gemm"))
        assert {row["physical_memory_component_id"] for row in qk.metadata["memory_accesses"]} == {"gddr0"}
    else:
        materialize = [task for task in tasks if task.metadata.get("event_kind") == "kv_materialize"
                       and task.metadata.get("phase") == "gpu_memory"]
        assert len(materialize) == 2
        for task in materialize:
            direct = task.metadata["phase_metadata"]["direct_memory_access"]
            assert direct["component_id"] == cache
            assert direct["read_bytes"] == 0
            assert direct["write_bytes"] == row_bytes
            assert direct["local_read_bytes"] == 2 * row_bytes
            assert direct["local_write_bytes"] == 0
        assert not any(task.metadata.get("event_kind") == "kv_store" for task in tasks)
        assert not any(task.demands for task in tasks if task.metadata.get("event_kind") == "kv_read")
