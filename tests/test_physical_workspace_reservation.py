from heterollm_sim.contracts import TaskSpec, TaskCategory
from heterollm_sim.workspace_memory import physical_workspace_bound
from dataclasses import replace
from types import SimpleNamespace


def task(name, dependencies=(), buffers=()):
    return TaskSpec(name, "cohort-0", name, TaskCategory.COMPUTE,
        dependencies=dependencies, metadata={
            "physical_owner": "hbm.memory",
            "physical_memory_config": {"burst_bytes": 1},
            "memory_accesses": tuple({"buffer_id": key, "byte_count": size,
                "buffer_size_bytes": size, "allocation_generation": generation,
                "operation": "read"} for key, size, generation in buffers)})


def test_parallel_branches_overlap_but_serial_regions_reuse_workspace():
    tasks = (task("start"), task("a", ("start",), (("a", 30, 1),)),
        task("b", ("start",), (("b", 50, 1),)), task("join", ("a", "b")),
        task("c", ("join",), (("c", 60, 1),)))
    audit = physical_workspace_bound(tasks)
    assert audit["bytes_by_owner"] == {"hbm.memory": 80}


def test_buffer_crossing_barrier_stays_live_until_its_last_user():
    tasks = (task("start"), task("a", ("start",), (("shared", 30, 1),)),
        task("join", ("a",)),
        task("c", ("join",), (("shared", 30, 1), ("c", 60, 1))))
    assert physical_workspace_bound(tasks)["bytes_by_owner"] == {"hbm.memory": 90}


def test_independent_future_root_does_not_create_a_false_serial_cut():
    tasks = (task("start"), task("a", ("start",), (("a", 30, 1),)),
        task("independent", (), (("b", 50, 1),)),
        task("join", ("a", "independent")))
    assert physical_workspace_bound(tasks)["bytes_by_owner"] == {"hbm.memory": 80}


def test_static_weights_and_persistent_kv_are_not_workspace():
    tasks = (task("start"), task("a", ("start",), (
        ("weights", 100000, 0), ("kv", 100000, 0), ("activation", 60, 1))))
    assert physical_workspace_bound(tasks)["bytes_by_owner"] == {"hbm.memory": 60}


def test_skip_edge_prevents_premature_workspace_reuse():
    tasks = (task("start"), task("a", ("start",), (("a", 30, 1),)),
        task("b", ("a",), (("b", 50, 1),)),
        task("c", ("a",), (("c", 60, 1),)), task("join", ("b", "c")))
    assert physical_workspace_bound(tasks)["bytes_by_owner"] == {"hbm.memory": 110}


def test_dangling_parallel_branch_cannot_retire_at_an_unrelated_barrier():
    tasks = (task("start"), task("long", ("start",), (("long", 80, 1),)),
        task("short", ("start",)), task("barrier", ("short",)),
        task("later", ("barrier",), (("later", 60, 1),)))
    assert physical_workspace_bound(tasks)["bytes_by_owner"] == {"hbm.memory": 140}


def test_redundant_earlier_dependency_does_not_hide_a_real_barrier():
    tasks = (task("start"), task("a", ("start",), (("a", 80, 1),)),
        task("barrier", ("a",)),
        task("later", ("start", "barrier"), (("later", 60, 1),)))
    assert physical_workspace_bound(tasks)["bytes_by_owner"] == {"hbm.memory": 80}


def test_tasklocal_buffers_in_parallel_tasks_are_distinct_allocations():
    tasks = (task("start"), task("a", ("start",), (("@tasklocal.output", 30, 1),)),
        task("b", ("start",), (("@tasklocal.output", 50, 1),)), task("join", ("a", "b")))
    assert physical_workspace_bound(tasks)["bytes_by_owner"] == {"hbm.memory": 80}


def test_nested_alias_extends_original_buffer_lifetime_without_double_count():
    a = task("a", ("start",), (("root", 80, 1),))
    view = task("view", ("a",), (("view", 40, 1),))
    view = replace(view, metadata={**view.metadata, "memory_accesses": (
        {**view.metadata["memory_accesses"][0], "alias_of": "root"},)})
    final = task("final", ("view",), (("nested", 20, 1), ("other", 60, 1)))
    final = replace(final, metadata={**final.metadata, "memory_accesses": (
        {**final.metadata["memory_accesses"][0], "alias_of": "view"}, final.metadata["memory_accesses"][1])})
    assert physical_workspace_bound((task("start"), a, view, final))["bytes_by_owner"] == {"hbm.memory": 140}


def test_workspace_audit_cache_is_owned_and_changes_with_typed_input(monkeypatch):
    from heterollm_sim import workspace_memory as w
    calls = []
    monkeypatch.setattr(w, "_device_workspace_reservation", lambda scenario:
        calls.append(scenario) or {"bound": [scenario["size"]]})
    scenario = {"unit_test_workspace": "isolated", "size": 11}
    first = w.device_workspace_reservation(scenario)
    first["bound"][0] = -1
    assert w.device_workspace_reservation(scenario) == {"bound": [11]}
    scenario["size"] = 12
    assert w.device_workspace_reservation(scenario) == {"bound": [12]}
    assert len(calls) == 2


def test_workspace_cache_ignores_only_presentation_mapping_status(monkeypatch):
    from copy import deepcopy
    from heterollm_sim import workspace_memory as w
    from heterollm_sim.reference import build_llama_default_scenario
    calls = []
    # This test exercises the real typed-input serialization without lowering.
    from collections import OrderedDict
    monkeypatch.setattr(w, "_AUDIT_CACHE", OrderedDict())
    monkeypatch.setattr(w, "_device_workspace_reservation", lambda scenario:
        calls.append(scenario) or {"bound": 64})
    scenario = build_llama_default_scenario()
    metadata = deepcopy(scenario.placement.metadata)
    metadata["ui"] = {"mapping_stale": True, "mapping_stale_reason": "model changed",
                      "selected_view": "model"}
    scenario = replace(scenario, placement=replace(scenario.placement, metadata=metadata))
    assert w.device_workspace_reservation(scenario) == {"bound": 64}
    cleared = deepcopy(metadata)
    cleared["ui"]["mapping_stale"] = False
    cleared["ui"].pop("mapping_stale_reason")
    validated = replace(scenario, placement=replace(scenario.placement, metadata=cleared))
    assert w.device_workspace_reservation(validated) == {"bound": 64}
    assert len(calls) == 1
    assert scenario.placement.metadata["ui"]["mapping_stale"] is True
    assert "mapping_stale_reason" in scenario.placement.metadata["ui"]
    other_ui = deepcopy(cleared)
    other_ui["ui"]["selected_view"] = "hardware"
    w.device_workspace_reservation(replace(validated,
        placement=replace(validated.placement, metadata=other_ui)))
    decision = deepcopy(cleared)
    decision["control_plane"] = {"input_fingerprint": "different actual decision"}
    w.device_workspace_reservation(replace(validated,
        placement=replace(validated.placement, metadata=decision)))
    w.device_workspace_reservation(replace(validated,
        workload=replace(validated.workload, metadata={**validated.workload.metadata,
            "source_contract": "different execution contract"})))
    assert len(calls) == 4


def test_mtp_capacity_analysis_lowers_maximum_verifier_and_acceptance_endpoints(monkeypatch):
    from heterollm_sim import planner
    from heterollm_sim import workspace_memory as w
    from heterollm_sim.reference import build_llama_default_scenario
    from heterollm_sim.ir import MTPPolicy, RequestSpec
    from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
    from contextlib import nullcontext
    scenario = build_llama_default_scenario()
    scenario = replace(scenario, llama_cpp_config=LlamaCppRuntimeConfig(batch=4, ubatch=4),
        workload=replace(scenario.workload, mtp=MTPPolicy(candidate_tokens=8)))
    monkeypatch.setattr(planner, "materialize_requests", lambda _: (RequestSpec("request", 0, 4, 12),))
    monkeypatch.setattr(planner, "_compilation_scope", lambda _: nullcontext())
    examples = []
    monkeypatch.setattr(planner, "_lower_serving_cohort", lambda scenario, cohort:
        examples.append(cohort) or SimpleNamespace(schedule=SimpleNamespace(tasks=(task("start"),))))
    monkeypatch.setattr(planner, "_promote_physical_allocation_extents", lambda tasks: tasks)
    w._device_workspace_reservation(scenario)
    mtp = [example for example in examples if example.kind == "mtp"]
    assert len(mtp) == 2
    assert {example.items[0].kv_append_tokens for example in mtp} == {1, 4}
    assert all(example.items[0].token_count == example.items[0].verifier_tokens == 4 for example in mtp)
    assert all(example.items[0].draft_tokens == 3 for example in mtp)
    assert examples[0].kind == "prefill"


def test_real_frontend_arena_placement_rebinds_control_plane_validation():
    import json
    from pathlib import Path
    from heterollm_sim.model_presets import materialize_model_payload
    from heterollm_sim.web import scenario_or_http_error, validation_payload, scenario_to_payload
    from heterollm_sim.llama_scenario import prepare_llama_scenario
    path = Path(__file__).resolve().parents[1] / "docs/frontend_native_validation_2026-10-07/ui_before_fix_physical_io_b200_3hbf_qwen3_0_6b_submission.json"
    payload = json.loads(path.read_text(encoding="utf-8"))["scenario"]
    payload["model"] = materialize_model_payload("qwen3-32b")
    payload["workload"]["prompt_tokens"] = 512
    payload["workload"]["output_tokens"] = 1
    payload["workload"]["requests"][0].update(prompt_tokens=512, output_tokens=1)
    payload["placement"]["model_name"] = payload["model"]["name"]
    for name in ("op_to_component", "tensor_to_component", "tensor_bytes"):
        payload["placement"][name] = {}
    payload["placement"]["metadata"].pop("control_plane", None)
    payload["placement"]["parallel"]["layer_to_stage"] = {}
    scenario = scenario_or_http_error(payload)
    assert scenario.placement.metadata["physical_workspace_reservation"]["physical_address_partitions"]
    validation = validation_payload(payload)
    assert validation["valid"], validation
    assert not validation["mapping_stale"]
    assert validation["input_fingerprint"] == validation["current_input_fingerprint"]
    assert prepare_llama_scenario(scenario) is scenario
    roundtrip_payload = scenario_to_payload(scenario)
    roundtrip = scenario_or_http_error(roundtrip_payload)
    assert roundtrip.placement.tensor_to_component == scenario.placement.tensor_to_component
    for key in ("total_allocated_bytes", "workspace_reserved_bytes"):
        assert roundtrip.placement.metadata["llama_device_memory_policy"][key] == scenario.placement.metadata["llama_device_memory_policy"][key]
    validation = validation_payload(roundtrip_payload)
    assert validation["valid"] and not validation["mapping_stale"], validation
    roundtrip_payload["workload"]["prompt_tokens"] = 1
    roundtrip_payload["workload"]["output_tokens"] = 2
    roundtrip_payload["workload"]["requests"][0].update(prompt_tokens=1, output_tokens=2)
    smaller = scenario_or_http_error(roundtrip_payload)
    policy = smaller.placement.metadata["llama_device_memory_policy"]
    assert policy["workspace_reserved_bytes"]["hbm0"] >= 57671680
    assert policy["total_allocated_bytes"]["hbm0"] <= policy["capacity_bytes"]["hbm0"]
