"""Runtime transactions and strict native source-assisted admission."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.cuda_graph_lifecycle import SOURCE_REVISION
from heterollm_sim.cuda_graph_contract import build_cuda_graph_contract
from heterollm_sim.ir import ComponentSpec, HardwareSpec, RequestSpec, WorkloadSpec
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
from heterollm_sim.cuda_graph_serving import (
    CudaGraphServingRuntime, _chain_descriptor, active_cuda_graph_cohort,
    cuda_graph_topology_descriptor,
    _phase_structures,
)


def structure():
    return {"nodes": [{"id": "a", "type": 0, "function": "fn", "grid": [1, 1, 1],
                       "block": [32, 1, 1], "shared_bytes": 0}], "edges": [], "edge_data": []}


def scenario(tmp_path):
    labels = ("model_warmup", "measured:0:prefill", "measured:0:decode:1")
    dry = [{"kind": "node_property", "id": 0, "bytes": "1122"}]
    capture = []
    for index, label in enumerate(labels):
        row = {"kind": "snapshot", "source_revision": SOURCE_REVISION,
               "label": label, "device": 0, "context_id": "ctx", "graph_key": "key",
               "graph_uid": 1, "n_nodes": 1, "compatible": True,
               "dry_run": True, "node_property_refs": [0], "call_id": index}
        dry.append(row)
        capture.append({**row, "dry_run": False, "capture_only": True})
        capture.append({"kind": "cuda_structure", "call_id": index,
                        "capture_only": True, "source_revision": SOURCE_REVISION, **structure()})
    result = SimpleNamespace(llama_cpp_config=LlamaCppRuntimeConfig(policy="generic"),
        model={"name": "model", "graph": {"dtype": "f16"}}, placement={"parallel": 1},
        hardware=HardwareSpec("hardware", (ComponentSpec("gpu", "gpu", cost_profile_id="gpu"),), ()),
        component_profiles={"gpu": {"gpu": SimpleNamespace(kernel_model=SimpleNamespace(
            graph_enabled=True, hardware_id="gpu", runtime_id="cuda", architecture="test"))}},
        workload=WorkloadSpec("workload", requests=(RequestSpec("startup", 0, 2, 1),
            RequestSpec("req", 0, 512, 2)), metadata={"cuda_graph_structural_program": {
            "schema": "heterollm.cuda-graph-source-program/v1", "graph_enabled": True,
            "dry_program_path": str(tmp_path / "dry"), "capture_program_path": str(tmp_path / "capture"),
            "request_prefixes": {"startup": "model_warmup", "req": "measured:0"},
            "request_stages": {"startup": "model_warmup", "req": "request"}}}))
    # The source-bound initialization is ordered before ordinary requests.
    result.workload = replace(result.workload, requests=(RequestSpec("startup", 0, 2, 1),
        RequestSpec("req", 1, 512, 2)))
    contract = build_cuda_graph_contract(result)
    result.workload.metadata["cuda_graph_structural_program"]["contract"] = contract
    for name, rows in (("dry", dry), ("capture", capture)):
        (tmp_path / name).write_text("\n".join(json.dumps(row) for row in (
            {"kind": "source_contract", "contract": contract}, *rows)), encoding="utf-8")
    return result


def cohort(request_id, phase="prefill", cursor=1):
    return SimpleNamespace(items=(SimpleNamespace(request_id=request_id, phase=phase,
        completion_cursor=2 if request_id == "startup" else cursor,
        token_count=2 if request_id == "startup" else 512, context_tokens=0),))


def tasks():
    return (TaskSpec("launch", "req", "launch", TaskCategory.COMPUTE,
        demands=(ResourceDemand("gpu.frontend", 1),), metadata={"phase": "kernel_launch",
            "target_component": "gpu", "operator_invocation_group_id": "g"}),
        TaskSpec("body", "req", "body", TaskCategory.COMPUTE, dependencies=("launch",),
            demands=(ResourceDemand("dram", 100, bytes_moved=1024),), metadata={"phase": "gpu_compute",
                "target_component": "gpu", "operator_invocation_group_id": "g"}))


def test_runtime_warmup_is_explicit_and_commit_occurs_after_success(tmp_path):
    runtime = CudaGraphServingRuntime(scenario(tmp_path))
    with pytest.raises(ValueError, match="warmup cannot be skipped"):
        with runtime.scope(cohort("req"), 0):
            pass
    with runtime.scope(cohort("startup"), 0) as prepared:
        assert active_cuda_graph_cohort() is prepared
        result = prepared.bind(tasks(), ({"group_id": "g", "request_ids": ("startup",)},))
        assert not result[0].metadata["cuda_graph_captured"]
        assert runtime.lifecycle.state.revision == 0
    assert active_cuda_graph_cohort() is None
    runtime.commit(prepared)
    assert runtime.lifecycle.state.revision == 1
    with runtime.scope(cohort("req"), 1000) as prepared:
        result = prepared.bind(tasks(), ({"group_id": "g", "request_ids": ("req",)},))
        assert result[0].metadata["cuda_graph_captured"]
        assert result[0].metadata["cuda_graph_node_count_basis"] == "native_capture_only_structural_compiler"
    runtime.commit(prepared)
    with runtime.scope(cohort("req", "decode", 2), 2000) as prepared:
        prepared.bind(tasks(), ({"group_id": "g", "request_ids": ("req",)},))
        assert prepared.transition.events == ("replay_submit",)
    runtime.commit(prepared)
    assert runtime.summary()["remaining_compiled_invocations"] == 0
    runtime.assert_complete()


def test_failed_runtime_does_not_reuse_prepared_capture(tmp_path):
    runtime = CudaGraphServingRuntime(scenario(tmp_path))
    with pytest.raises(ValueError, match="not fully executed"):
        runtime.assert_complete()
    with runtime.scope(cohort("startup"), 0) as prepared:
        prepared.bind(tasks(), ({"group_id": "g", "request_ids": ("startup",)},))
    runtime.failed()
    with pytest.raises(ValueError, match="stale"):
        runtime.commit(prepared)


@pytest.mark.parametrize("mutation", ["model", "dtype", "prompt", "context", "batch", "kv", "prefix"])
def test_runtime_rejects_structures_compiled_for_different_inputs(tmp_path, mutation):
    value = scenario(tmp_path)
    if mutation == "model":
        value.model["name"] = "different_model"
    elif mutation == "dtype":
        value.model["graph"]["dtype"] = "q8_0"
    elif mutation == "prompt":
        value.workload = replace(value.workload, requests=(value.workload.requests[0],
            replace(value.workload.requests[1], prompt_tokens=511)))
    elif mutation in {"context", "batch"}:
        value.llama_cpp_config = replace(value.llama_cpp_config, **{mutation: 1024})
    elif mutation == "kv":
        value.llama_cpp_config = replace(value.llama_cpp_config, kv_type_k="f32")
    else:
        value.workload.metadata["cuda_graph_structural_program"]["request_prefixes"]["req"] = "different"
    with pytest.raises(ValueError, match="compilation contract mismatch"):
        CudaGraphServingRuntime(value)


@pytest.mark.parametrize("kind", ["dry", "capture"])
def test_program_contract_cannot_relabel_an_old_file(tmp_path, kind):
    value = scenario(tmp_path)
    program = value.workload.metadata["cuda_graph_structural_program"]
    wrong_file = tmp_path / (kind + "_other")
    records = [json.loads(line) for line in (tmp_path / kind).read_text().splitlines()]
    records[0]["contract"]["model"]["name"] = "other_model"
    wrong_file.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    program[kind + "_program_path"] = str(wrong_file)
    with pytest.raises(ValueError, match="compilation contract mismatch"):
        CudaGraphServingRuntime(value)


def test_missing_producer_contract_is_not_silently_accepted(tmp_path):
    value = scenario(tmp_path)
    path = tmp_path / "dry"
    path.write_text("\n".join(path.read_text().splitlines()[1:]), encoding="utf-8")
    with pytest.raises(ValueError, match="no producer compilation contract"):
        CudaGraphServingRuntime(value)


@pytest.mark.parametrize("program_mode,profile_mode", [(False, True), (True, False)])
def test_graph_mode_must_match_before_any_cohort_executes(tmp_path, program_mode, profile_mode):
    value = scenario(tmp_path)
    value.workload.metadata["cuda_graph_structural_program"]["graph_enabled"] = program_mode
    value.component_profiles["gpu"]["gpu"].kernel_model.graph_enabled = profile_mode
    with pytest.raises(ValueError, match="Graph mode contradicts"):
        CudaGraphServingRuntime(value)


def test_initialization_sampling_rule_requires_explicit_stage_and_shape(tmp_path):
    value = scenario(tmp_path)
    program = value.workload.metadata["cuda_graph_structural_program"]
    program["request_stages"]["startup"] = "request"
    with pytest.raises(ValueError, match="ordinary CUDA request"):
        CudaGraphServingRuntime(value)
    program["request_stages"]["startup"] = "model_warmup"
    value.workload = replace(value.workload, requests=(
        replace(value.workload.requests[0], prompt_tokens=512), value.workload.requests[1]))
    with pytest.raises(ValueError, match="two-token"):
        CudaGraphServingRuntime(value)


def test_cohort_cannot_borrow_initialization_stage_for_a_different_request(tmp_path):
    runtime = CudaGraphServingRuntime(scenario(tmp_path))
    with runtime.scope(cohort("startup"), 0) as prepared:
        assert prepared.stage == "model_warmup"
        with pytest.raises(ValueError, match="active request"):
            prepared.bind(tasks(), ({"group_id": "g", "request_ids": ("req",)},))


def test_same_structural_contract_supports_matched_off_mode(tmp_path):
    value = scenario(tmp_path)
    value.workload.metadata["cuda_graph_structural_program"]["graph_enabled"] = False
    value.component_profiles["gpu"]["gpu"].kernel_model.graph_enabled = False
    runtime = CudaGraphServingRuntime(value)
    for item in (cohort("startup"), cohort("req"), cohort("req", "decode", 2)):
        with runtime.scope(item, 0) as prepared:
            prepared.bind(tasks(), ({"group_id": "g", "request_ids": (item.items[0].request_id,)},))
            assert prepared.transition.events == ("ordinary_submit",)
        runtime.commit(prepared)
    runtime.assert_complete()


def test_off_independent_costs_cannot_disappear_with_removed_program(tmp_path):
    value = scenario(tmp_path)
    value.workload.metadata.pop("cuda_graph_structural_program")
    profile = value.component_profiles["gpu"]["gpu"].kernel_model
    profile.graph_enabled = False
    profile.runtime_calibration = object()
    with pytest.raises(ValueError, match="both Graph modes"):
        CudaGraphServingRuntime(value)


def test_frontend_validation_reports_static_contract_mismatch_without_reading_structures(tmp_path, monkeypatch):
    from heterollm_sim import planner
    from pathlib import Path
    value = scenario(tmp_path)
    value.model["name"] = "a_different_model"
    def unexpected(*args, **kwargs):
        raise AssertionError("frontend contract validation must not read structural files")
    monkeypatch.setattr(Path, "open", unexpected)
    report = planner.validate_scenario(value)
    assert report.errors
    assert any("contract.model.name" in message for message in report.errors)


def test_static_admission_of_matching_contract_needs_no_structural_file_io(tmp_path, monkeypatch):
    from heterollm_sim.cuda_graph_contract import validate_cuda_graph_scenario_contract
    from pathlib import Path
    value = scenario(tmp_path)
    def unexpected(*args, **kwargs):
        raise AssertionError("static admission must not read traces")
    monkeypatch.setattr(Path, "open", unexpected)
    assert validate_cuda_graph_scenario_contract(value) == value.workload.metadata[
        "cuda_graph_structural_program"]["contract"]


def test_internal_placement_probe_exemption_cannot_admit_a_real_runtime(tmp_path):
    from heterollm_sim.cuda_graph_contract import validate_cuda_graph_scenario_contract
    value = scenario(tmp_path)
    value.workload.metadata["_control_plane_placement_validation"] = True
    value.model["name"] = "changed"
    with pytest.raises(ValueError, match="compilation contract mismatch"):
        validate_cuda_graph_scenario_contract(value, allow_placement_probe=True)
    value.workload = replace(value.workload, requests=(), request_count=1,
        prompt_tokens=1, output_tokens=0)
    assert validate_cuda_graph_scenario_contract(value, allow_placement_probe=True) is None
    with pytest.raises(ValueError, match="every explicit request ID"):
        CudaGraphServingRuntime(value)


def test_real_frontend_normalize_and_validate_preserve_compilation_contract():
    from pathlib import Path
    from heterollm_sim.cuda_graph_contract import validate_cuda_graph_scenario_contract
    from heterollm_sim.web import validation_payload, scenario_or_http_error, scenario_to_payload
    path = Path(__file__).parents[1] / "docs/cuda_graph_validation_2026-10-08/scenario_qwen3_0_6b_f16_graph_on.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    program = payload["workload"]["metadata"]["cuda_graph_structural_program"]
    # Use the producer's actual contract; a frontend regression must not make
    # an obsolete fixture pass by rebuilding/relabeling its evidence in-test.
    expected = program["contract"]
    assert expected["placement"]["op_to_component"]
    # The actual browser rebuilds the tensor directory and writes layout into
    # graph.attributes.ui. These edits cannot invalidate compiled execution.
    payload["model"]["graph"]["tensors"].reverse()
    payload["model"]["graph"]["attributes"]["ui"] = {
        "positions": {"layer_0": {"x": 20, "y": 40}}, "viewport": {"scale": 0.5}}
    for _ in range(2):
        result = validation_payload(payload)
        assert result["valid"], result["errors"]
        normalized = scenario_or_http_error(payload)
        assert validate_cuda_graph_scenario_contract(normalized) == expected
        payload = scenario_to_payload(normalized)
    changed_mapping = dict(normalized.placement.op_to_component)
    mapped_operator = next(key for key, component in changed_mapping.items() if component == "gpu0")
    changed_mapping[mapped_operator] = "cpu0"
    remapped = replace(normalized, placement=replace(normalized.placement,
        op_to_component=changed_mapping))
    with pytest.raises(ValueError, match="contract.placement.op_to_component"):
        validate_cuda_graph_scenario_contract(remapped)
    payload["workload"]["requests"][-1]["prompt_tokens"] = 511
    result = validation_payload(payload)
    assert not result["valid"]
    assert any("contract" in issue["message_en"] for issue in result["errors"]["scenario"])


def test_contract_model_semantics_ignore_only_ui_and_tensor_directory_order():
    from copy import deepcopy
    from heterollm_sim.cuda_graph_contract import require_matching_cuda_graph_contract
    model = {"name": "model", "metadata": {"ui": {"x": 1}, "architecture": "qwen3"},
        "graph": {"operators": [{"operator_id": "a"}, {"operator_id": "b"}],
            "tensors": [{"tensor_id": "z", "dtype": "f16", "shape": [4, 8]},
                        {"tensor_id": "a", "dtype": "q8_0", "shape": [8, 16]}],
            "attributes": {"ui": {"positions": {}}, "hidden_size": 8}}}
    original = deepcopy(model)
    display = deepcopy(model)
    display["graph"]["tensors"].reverse()
    display["graph"]["attributes"]["ui"] = {"positions": {"a": {"x": 0, "y": 10}}}
    display["graph"]["source_operators"] = []
    display["graph"]["sub_operators"] = []
    require_matching_cuda_graph_contract({"model": original}, {"model": display}, source="test")
    assert original == model  # Comparisons must not mutate producer evidence.
    for mutation in ("operators", "dtype", "shape", "attribute", "tensor_id"):
        changed = deepcopy(display)
        if mutation == "operators": changed["graph"]["operators"].reverse()
        elif mutation in {"dtype", "shape", "tensor_id"}:
            changed["graph"]["tensors"][0][mutation] = {
                "dtype": "f32", "shape": [16, 8], "tensor_id": "different"}[mutation]
        else: changed["graph"]["attributes"]["hidden_size"] = 16
        with pytest.raises(ValueError, match="compilation contract mismatch"):
            require_matching_cuda_graph_contract({"model": original}, {"model": changed}, source="test")


def test_browser_numeric_encoding_and_placement_layout_do_not_change_contract():
    from copy import deepcopy
    from heterollm_sim.cuda_graph_contract import require_matching_cuda_graph_contract
    producer = {"placement": {"metadata": {"policy": "llama_cpp"},
        "kv_policy": {"offload_ratio": 0.0}, "op_to_component": {"matmul": "gpu0"}},
        "workload": {"enabled": True, "arrival_ns": 0.0}}
    browser = deepcopy(producer)
    browser["placement"]["metadata"]["ui"] = {"positions": {"gpu0": {"x": 25}}}
    browser["placement"]["kv_policy"]["offload_ratio"] = 0
    browser["workload"]["arrival_ns"] = 0
    require_matching_cuda_graph_contract(producer, browser, source="browser")
    for mutation in ("boolean", "ratio", "mapping", "policy"):
        changed = deepcopy(browser)
        if mutation == "boolean": changed["workload"]["enabled"] = 1
        elif mutation == "ratio": changed["placement"]["kv_policy"]["offload_ratio"] = 0.5
        elif mutation == "mapping": changed["placement"]["op_to_component"]["matmul"] = "cpu0"
        else: changed["placement"]["metadata"]["policy"] = "generic"
        with pytest.raises(ValueError, match="compilation contract mismatch"):
            require_matching_cuda_graph_contract(producer, changed, source="browser")


def test_multiple_backend_splits_or_gpu_targets_do_not_merge(tmp_path):
    runtime = CudaGraphServingRuntime(scenario(tmp_path))
    with runtime.scope(cohort("startup"), 0) as prepared:
        with pytest.raises(ValueError, match="one physical ubatch"):
            prepared.bind(tasks(), ({}, {}))
        extra = replace(tasks()[0], task_id="launch2", metadata={
            **tasks()[0].metadata, "target_component": "gpu2"})
        with pytest.raises(ValueError, match="exactly one GPU"):
            prepared.bind((*tasks(), extra), ({"group_id": "g", "request_ids": ("startup",)},))


def test_device_interior_markers_are_bound_without_crossing_cpu_embedding(tmp_path, monkeypatch):
    from heterollm_sim import planner
    monkeypatch.setattr(planner, "_parallel_plan", lambda _: SimpleNamespace(ranks=(
        SimpleNamespace(component_id="gpu", memory_component_id="gddr"),)))
    runtime = CudaGraphServingRuntime(scenario(tmp_path))
    group = {"operator_invocation_group_id": "g"}
    leading = TaskSpec("leading", "req", "leading", TaskCategory.COMMUNICATION, metadata=group)
    cpu = TaskSpec("cpu", "req", "cpu", TaskCategory.COMPUTE, dependencies=("leading",),
        demands=(ResourceDemand("cpu.memory", 10),), metadata={**group, "target_component": "cpu"})
    launch, body = tasks()
    launch = replace(launch, dependencies=("cpu",))
    marker = TaskSpec("marker", "req", "marker", TaskCategory.SYNCHRONIZATION,
        dependencies=("launch",), metadata=group)
    kv = TaskSpec("kv", "req", "kv", TaskCategory.COMMUNICATION, dependencies=("marker",),
        metadata={**group, "target_component": "gddr", "source_component": "gddr", "event_kind": "kv_read",
                  "resource_accounting": "included_in_attention_kernel"})
    body = replace(body, dependencies=("kv",))
    with runtime.scope(cohort("startup"), 0) as prepared:
        result = prepared.bind((leading, cpu, launch, marker, kv, body),
                               ({"group_id": "g", "request_ids": ("startup",)},))
    owner = result[2]
    assert owner.metadata["cuda_graph_member_task_ids"] == ("launch", "marker", "kv", "body")
    assert "cuda_graph_lifecycle" not in result[0].metadata
    assert "cuda_graph_lifecycle" not in result[1].metadata
    assert owner.dependencies == ("cpu",)
    assert owner.metadata["cuda_graph_structure_id"] in runtime.structures_registry


def test_foreign_host_work_inside_gpu_body_remains_rejected(tmp_path):
    runtime = CudaGraphServingRuntime(scenario(tmp_path))
    launch, body = tasks()
    cpu = TaskSpec("cpu", "req", "cpu", TaskCategory.COMPUTE, dependencies=("launch",),
        demands=(ResourceDemand("cpu.memory", 10),), metadata={
            "operator_invocation_group_id": "g", "target_component": "cpu"})
    body = replace(body, dependencies=("cpu",))
    with runtime.scope(cohort("startup"), 0) as prepared:
        with pytest.raises(ValueError, match="host dependency split"):
            prepared.bind((launch, cpu, body), ({"group_id": "g", "request_ids": ("startup",)},))


def recurrent_state_tasks():
    launch, body = tasks()
    state = TaskSpec("state", "req", "linear_state.write", TaskCategory.MEMORY,
        dependencies=("body",), demands=(ResourceDemand("gddr.fabric", 80, bytes_moved=4096),),
        metadata={"operator_invocation_group_id": "g", "event_kind": "linear_state_write",
            "source_component": "gpu", "target_component": "gddr",
            "direct_memory_component": "gddr", "resource_accounting": "direct_memory_access",
            "resource_transfer_bytes": 0, "access_kind": "WRITE",
            "physical_memory_config": {"kind": "GDDR"}, "physical_owner": "gddr.fabric",
            "memory_access": {"operation": "write", "physical_owner": "gddr.fabric", "byte_count": 4096}})
    next_launch = replace(launch, task_id="launch2", dependencies=("state",))
    next_body = replace(body, task_id="body2", dependencies=("launch2",))
    return launch, body, state, next_launch, next_body


def test_local_recurrent_state_write_is_graph_body_with_physical_cost_intact(tmp_path, monkeypatch):
    from heterollm_sim import planner
    monkeypatch.setattr(planner, "_parallel_plan", lambda _: SimpleNamespace(ranks=(
        SimpleNamespace(component_id="gpu", memory_component_id="gddr"),)))
    runtime = CudaGraphServingRuntime(scenario(tmp_path))
    original = recurrent_state_tasks()
    for request in ("startup", "req"):
        with runtime.scope(cohort(request), 0) as prepared:
            result = prepared.bind(original, ({"group_id": "g", "request_ids": (request,)},))
        runtime.commit(prepared)
        assert result[0].metadata["cuda_graph_member_task_ids"] == tuple(task.task_id for task in original)
        assert result[2].metadata["target_component"] == "gddr"
        assert result[2].metadata["cuda_graph_device"] == "gpu"
        assert result[2].metadata["cuda_graph_captured"] == (request == "req")
        assert tuple(task.demands for task in result) == tuple(task.demands for task in original)
        assert tuple(task.dependencies for task in result) == tuple(task.dependencies for task in original)
        assert result[2].metadata["memory_access"] == original[2].metadata["memory_access"]


@pytest.mark.parametrize("mutation", ["host_storage", "host_source", "missing_physics", "wrong_owner", "transfer", "cpu_demand"])
def test_physical_graph_member_requires_local_memory_and_device_access(tmp_path, monkeypatch, mutation):
    from heterollm_sim import planner
    monkeypatch.setattr(planner, "_parallel_plan", lambda _: SimpleNamespace(ranks=(
        SimpleNamespace(component_id="gpu", memory_component_id="gddr"),)))
    original = list(recurrent_state_tasks())
    state = original[2]
    meta = dict(state.metadata)
    if mutation == "host_storage":
        meta.update(target_component="hostmem", direct_memory_component="hostmem")
    elif mutation == "host_source":
        meta["source_component"] = "cpu"
    elif mutation == "missing_physics":
        meta.pop("physical_memory_config")
    elif mutation == "wrong_owner":
        meta["memory_access"] = {**meta["memory_access"], "physical_owner": "hostmem.fabric"}
    elif mutation == "transfer":
        meta["resource_transfer_bytes"] = 4096
    else:
        state = replace(state, demands=(ResourceDemand("cpu.memory", 80, bytes_moved=4096),))
    original[2] = replace(state, metadata=meta)
    runtime = CudaGraphServingRuntime(scenario(tmp_path))
    with runtime.scope(cohort("startup"), 0) as prepared:
        with pytest.raises(ValueError, match="host dependency split"):
            prepared.bind(tuple(original), ({"group_id": "g", "request_ids": ("startup",)},))


def test_nondefault_edges_and_memcpy_nodes_cannot_use_kernel_chain_calibration():
    graph = structure()
    graph["nodes"].append({**graph["nodes"][0], "id": "b"})
    graph["edges"] = [["a", "b"]]
    graph["edge_data"] = ["0100010000000000"]
    with pytest.raises(ValueError, match="dependency edge types"):
        _chain_descriptor(graph)
    graph["edge_data"] = ["0000000000000000"]
    graph["nodes"][1]["type"] = 1
    with pytest.raises(ValueError, match="non-kernel"):
        _chain_descriptor(graph)


def test_typed_chain_key_preserves_memcpy_geometry_and_programmatic_edges():
    kernel = {**structure()["nodes"][0], "cooperative": 0}
    copy = {"src_memory_type": 2, "dst_memory_type": 2, "width_bytes": 64,
            "height": 1, "depth": 1, "src_pitch": 0, "dst_pitch": 0,
            "src_context": "ctx", "dst_context": "ctx",
            "src_device_ordinal": 0, "dst_device_ordinal": 0}
    graph = {"nodes": [kernel, {**kernel, "id": "b"},
                        {"id": "c", "type": 1, "copy": copy}, {**kernel, "id": "d"}],
             "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
             "edge_data": ["0100010000000000", "0000000000000000", "0000000000000000"]}
    result = cuda_graph_topology_descriptor(graph)
    assert result["topology"].startswith("typed_chain/v1:")
    assert "0100010000000000" in result["topology"]
    assert '["memcpy",2,2,64,1,1,0,0,' in result["topology"]
    assert result["update_compatibility_covered"]
    graph["nodes"][2]["copy"] = {**copy, "width_bytes": 128}
    assert cuda_graph_topology_descriptor(graph)["topology"] != result["topology"]


def test_recapture_prices_old_destruction_and_exact_old_to_new_update_pair():
    previous = {"topology": "typed:old", "node_count": 533}
    current = {"topology": "typed:new", "node_count": 425}
    phases = _phase_structures(("destroy_graph", "capture", "update_failure",
                              "destroy_exec", "instantiate", "first_launch_submit"), current, previous)
    assert phases["destroy_graph"] == phases["destroy_exec"] == previous
    assert phases["capture"] == phases["instantiate"] == phases["first_launch_submit"] == current
    assert phases["update_failure"]["previous"] == previous
    assert phases["update_failure"]["current"] == current
    initial = _phase_structures(("capture", "instantiate", "update"), current, None)
    assert initial["update"]["previous"] == initial["update"]["current"] == current
