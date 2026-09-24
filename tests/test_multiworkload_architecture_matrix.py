"""Matrix authoring/whitelist regressions; no GGUF or serving execution.

The build fixture stubs only fixed_reference for reference-model constructor
tests. I1 real mapping, HBF parameters and I4 use the ordinary planner with a
small hybrid model. No test certifies full serving execution.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import multiworkload_architecture_matrix as matrix
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.ir import LayerSpec, LinearAttentionSpec
from tests.model_helpers import model_from_layer_specs

OPTIONS = "/placement/metadata/control_plane/policy/options"
LOAD = ("test_small", 8, 4, 2)


def test_eager_admission_preserves_entire_declared_load(build):
    scenario = build("kv_hbm")
    assert scenario.placement.kv_policy.allocation_policy == "eager"
    assert len(scenario.workload.requests) == LOAD[3]
    assert all((r.prompt_tokens, r.output_tokens) == LOAD[1:3]
               for r in scenario.workload.requests)
    assert scenario.workload.scheduler.max_num_seqs == LOAD[3]
    assert not scenario.workload.scheduler.preemption_enabled


def test_p4_feasible_scratch_sweep_keeps_physical_capacity(build):
    group = next(g for g in matrix.GROUPS if g["id"] == "P4")
    assert group["values"] == [262144, 524288, 1048576]
    scenarios = [build(case) for case in group["cases"]]
    assert all(s.hardware.get_component("cim0").capacity_bytes == 3 * 1024**2
               for s in scenarios)
    for scenario in scenarios:
        profile = scenario.resolve_component_profile("cim0")
        assert profile.weight_capacity_bytes + profile.conversion_scratch_capacity_bytes <= 3 * 1024**2


def test_summary_handles_incomplete_null_metrics(tmp_path):
    workload = ("incomplete", 32768, 128, 16)
    group = matrix.GROUPS[0]
    for case in group["cases"]:
        matrix.write_json(tmp_path / "cells" / (workload[0] + "__" + case + ".json"),
            {"status": "CHECK_FAILED", "observation": {
                "metrics": {"ttft_ns": None, "tpot_ns": None, "e2e_ns": None},
                "summary": {"throughput": {"visible_output_tokens_per_s": 123.45}}}})
    rows = matrix.summarize(tmp_path, [group], [workload])
    assert not rows[0]["comparable"]
    assert all(p["metrics_ms"] == {} and p["throughput_tokens_per_s"] is None
               for p in rows[0]["points"])
    assert "CHECK_FAILED" in (tmp_path / "results.md").read_text(encoding="utf-8")


@pytest.fixture
def reference():
    return build_reference_scenario()


@pytest.fixture(autouse=True)
def forbid_external_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("These regressions must not load GGUF or execute serving")

    monkeypatch.setattr(matrix.q, "load_model", forbidden)
    monkeypatch.setattr(matrix.q, "execute_scenario", forbidden)
    monkeypatch.setattr(matrix.q, "run_scenario", forbidden)


@pytest.fixture
def build(reference, monkeypatch):
    def fixed_reference(scenario, component):
        options = deepcopy(scenario.placement.metadata["control_plane"]["policy"]["options"])
        options["operator_targets"] = dict(reference.placement.op_to_component)
        layers = [item.layer_id for item in reference.model._execution_view.layer_instances]
        options["kv_layer_targets"] = {layer: options["kv_cache_target"] for layer in layers}
        return matrix.q._constraints(scenario, options), {}

    monkeypatch.setattr(matrix.sweep, "fixed_reference", fixed_reference)
    return lambda variant: matrix.build_case(reference.model, variant, LOAD)[0]


@pytest.fixture
def hybrid_model():
    full = LayerSpec("layer-000", "dense", hidden_size=64, intermediate_size=128,
                     attention_heads=4, dtype="fp16")
    linear = replace(full, layer_id="layer-001", sequence_mixer="linear_attention",
                     linear_attention=LinearAttentionSpec(2, 4, 16, 16, 4))
    return model_from_layer_specs("matrix-small-hybrid", (full, linear))


def test_i1_real_planner_changes_only_kv_mapping(hybrid_model):
    # No fixed_reference stub: exercise the ordinary planner, never serving.
    left, right = (matrix.build_case(hybrid_model, case, LOAD)[0]
                   for case in ("kv_hbm", "kv_hbf"))
    assert matrix.differences(matrix.q.scenario_payload(left), matrix.q.scenario_payload(right)) == [
        OPTIONS + "/kv_cache_target"]
    before, after = (matrix.sweep.placement_mapping(matrix.sweep.solve_placement(s))
                     for s in (left, right))
    assert matrix.differences(before, after) == ["/tensor_to_component/kv_cache"]
    assert before["tensor_to_component"]["kv_cache"] == "hbm0"
    assert after["tensor_to_component"]["kv_cache"] == "hbf0"
    assert before["tensor_to_component"]["linear_state"] == after["tensor_to_component"]["linear_state"] == "hbm2"


@pytest.mark.parametrize("left,right,expected", [
    ({"a": {"b": 1}}, {"a": {"b": 2}}, ["/a/b"]),
    ({"a": [1, {"b": 2}]}, {"a": [1, {"b": 3}]}, ["/a/1/b"]),
    ({"a": [1]}, {"a": [1, 2]}, ["/a"]),
    ({"a": {}}, {"a": {"b": {}}}, ["/a/b"]),
    ({"a": {"b": {}}}, {"a": {}}, ["/a/b"]),
    ({"a": [1, {"b": 2}]}, {"a": [1, {"b": 2}]}, []),
])
def test_differences_reports_leaf_and_structural_changes(left, right, expected):
    assert matrix.differences(left, right) == expected


@pytest.mark.parametrize("prefix", ["dram", "cim"])
def test_shared_pair_changes_only_three_fabric_owners(build, prefix):
    left, right = (matrix.q.scenario_payload(build(prefix + suffix))
                   for suffix in ("_independent", "_shared"))
    expected = {"/hardware/metadata/phy_noc_mode"}
    selected = {"vertical_dram0", "vertical_dram1", "soc_cim_noc"}
    right_links = {link["link_id"]: link for link in right["hardware"]["links"]}
    for link in left["hardware"]["links"]:
        if link["link_id"] in selected:
            expected.add(f"/hardware/links/{link['link_id']}/metadata/physical_resource_owner")
            expected.add("/hardware/metadata/physical_resource_owners/link." + link["link_id"])
            assert right_links[link["link_id"]]["metadata"]["physical_resource_owner"] == "stack0.shared_phy_noc"
            assert right["hardware"]["metadata"]["physical_resource_owners"]["link." + link["link_id"]] == "stack0.shared_phy_noc"
    assert len(expected) == 7
    assert set(matrix.differences(left, right)) == expected
    assert all(matrix.allowed_change(path, "topology") for path in expected)
    assert right["hardware"]["metadata"]["phy_noc_mode"] == "shared"
    assert len(right["workload"]["requests"]) == LOAD[3]


@pytest.mark.parametrize("path", [
    "/hardware/links/vertical_dram0/bandwidth_gbps",
    "/hardware/links/vertical_dram0/latency_ns",
    "/hardware/components/dram0/read_bandwidth_gbps",
    "/profiles/components/host_memory/example-dram0/read_latency_ns",
    OPTIONS + "/operator_targets/op",
    "/placement/op_to_component/op",
    OPTIONS + "/weight_tensor_targets/weight",
    "/weights_resident",
    "/workload/scheduler/max_num_seqs",
    "/workload/requests/0/prompt_tokens",
])
def test_owner_only_whitelist_rejects_other_factors(path):
    assert not matrix.allowed_change(path, "topology")


def test_i1_same_hardware_weights_operators_and_workload_only_kv_moves(build):
    left, right = (matrix.q.scenario_payload(build(case)) for case in ("kv_hbm", "kv_hbf"))
    options_left = left["placement"]["metadata"]["control_plane"]["policy"]["options"]
    options_right = right["placement"]["metadata"]["control_plane"]["policy"]["options"]
    assert options_left["kv_cache_target"] == "hbm0"
    assert options_right["kv_cache_target"] == "hbf0"
    assert options_left["kv_layer_targets"]
    assert set(options_left["kv_layer_targets"].values()) == {"hbm0"}
    assert set(options_right["kv_layer_targets"].values()) == {"hbf0"}
    expected = {OPTIONS + "/kv_cache_target"} | {
        OPTIONS + "/kv_layer_targets/" + layer for layer in options_left["kv_layer_targets"]}
    assert set(matrix.differences(left, right)) == expected
    assert all(matrix.allowed_change(path, "placement") for path in expected)
    assert left["hardware"] == right["hardware"]
    assert options_left["linear_state_target"] == options_right["linear_state_target"] == "hbm2"
    assert set(options_left["weight_tensor_targets"].values()) == {"hbm1"}
    assert len(left["workload"]["requests"]) == LOAD[3]


@pytest.mark.parametrize("path", [
    OPTIONS + "/linear_state_target", OPTIONS + "/linear_state_layer_targets/layer",
    OPTIONS + "/weight_tensor_targets/weight", OPTIONS + "/operator_targets/op",
    "/placement/parallel/rank_mapping/0/memory_component_id", "/weights_resident",
    "/hardware/components/hbm0/capacity_bytes", "/workload/requests/0/output_tokens",
])
def test_i1_whitelist_rejects_non_kv_changes(path):
    assert not matrix.allowed_change(path, "placement")


@pytest.mark.parametrize("kind,field", [
    ("hbf_latency", "read_latency_ns"),
    ("hbf_outstanding", "max_outstanding_requests"),
])
def test_hbf_whitelist_accepts_only_actual_profile_and_metadata_owner(reference, kind, field):
    scenario = matrix.q.build_scenario(reference.model, "hbf_active_weights")
    hbf = scenario.hardware.get_component("hbf0")
    profile_path = f"/profiles/components/host_memory/{hbf.cost_profile_id}/{field}"
    assert matrix.allowed_change(profile_path, kind)
    assert matrix.allowed_change(f"/hardware/components/hbf0/metadata/{field}", kind)
    for profile in ("example-dram0", "example-dram1", "legacy-host-memory", hbf.cost_profile_id + "-other"):
        assert not matrix.allowed_change(f"/profiles/components/host_memory/{profile}/{field}", kind)
    for component in scenario.hardware.components:
        if component.component_id != "hbf0":
            path = f"/hardware/components/{component.component_id}/metadata/{field}"
            assert not matrix.allowed_change(path, kind), (component.component_id, path)
    assert not matrix.allowed_change(f"/hardware/links/gpu-hbf0/metadata/{field}", kind)
    assert not matrix.allowed_change(profile_path + "/extra", kind)


@pytest.mark.parametrize("variant,kind,field,value", [
    ("hbf_lat_1000", "hbf_latency", "read_latency_ns", 1000.0),
    ("hbf_lat_20000", "hbf_latency", "read_latency_ns", 20000.0),
    ("hbf_q_256", "hbf_outstanding", "max_outstanding_requests", 256),
    ("hbf_q_16384", "hbf_outstanding", "max_outstanding_requests", 16384),
])
def test_hbf_parameter_pair_builds_with_synchronized_profile_and_metadata(hybrid_model, variant, kind, field, value):
    base, changed = (matrix.build_case(hybrid_model, case, LOAD)[0]
                     for case in ("hbf_lat_10000", variant))
    left, right = matrix.q.scenario_payload(base), matrix.q.scenario_payload(changed)
    hbf = changed.hardware.get_component("hbf0")
    expected = {f"/profiles/components/host_memory/{hbf.cost_profile_id}/{field}",
                f"/hardware/components/hbf0/metadata/{field}"}
    assert set(matrix.differences(left, right)) == expected
    assert all(matrix.allowed_change(path, kind) for path in expected)
    assert getattr(changed.resolve_component_profile("hbf0"), field) == hbf.metadata[field] == value
    assert left["workload"] == right["workload"]
    assert len(right["workload"]["requests"]) == LOAD[3]


@pytest.mark.parametrize("collection,id_field", [("components", "component_id"), ("links", "link_id")])
def test_differences_uses_ids_not_list_positions(collection, id_field):
    first = {id_field: "first", "metadata": {"read_latency_ns": 100}}
    second = {id_field: "second", "metadata": {"read_latency_ns": 200}}
    left = {"hardware": {collection: [first, second]}}
    reordered = {"hardware": {collection: [deepcopy(second), deepcopy(first)]}}
    assert matrix.differences(left, reordered) == []
    reordered["hardware"][collection][0]["metadata"]["read_latency_ns"] = 300
    assert matrix.differences(left, reordered) == [f"/hardware/{collection}/second/metadata/read_latency_ns"]
    assert matrix.differences(left, {"hardware": {collection: [first]}}) == [f"/hardware/{collection}/second"]


@pytest.mark.parametrize("path", [
    "/hardware/links/cpu-gpu-pcie/metadata/physical_resource_owner",
    "/hardware/links/gpu-hbm7/metadata/physical_resource_owner",
    "/hardware/metadata/physical_resource_owners/link.cpu-gpu-pcie",
    "/hardware/metadata/physical_resource_owners/link.gpu-hbm7",
    "/hardware/components/dram0/metadata/memory_service_owner",
])
def test_topology_whitelist_rejects_unrelated_resource_owners(path):
    assert not matrix.allowed_change(path, "topology")


@pytest.mark.parametrize("path", [
    "/profiles/components/host_memory/example-dram1/read_latency_ns",
    "/profiles/components/host_memory/example-hbf-memory/read_latency_ns",
    "/hardware/components/dram1/metadata/read_latency_ns",
    "/hardware/components/hbf0/metadata/read_latency_ns",
    "/hardware/links/vertical_dram0/latency_ns",
])
def test_dram_latency_whitelist_rejects_other_owners(path):
    assert matrix.allowed_change("/profiles/components/host_memory/example-dram0/read_latency_ns", "dram_latency")
    assert not matrix.allowed_change(path, "dram_latency")


@pytest.mark.parametrize("batch", [1, 2])
def test_i4_flash_weights_share_only_four_links_without_changing_devices(hybrid_model, batch):
    group = next((g for g in matrix.GROUPS if g["id"] == "I4"), None)
    assert group is not None, "I4 remote Flash weight owner comparison must be registered"
    assert len(group["cases"]) == 2
    load = ("test_flash", 8, 4, batch)
    scenarios = [matrix.build_case(hybrid_model, case, load)[0] for case in group["cases"]]
    left, right = [matrix.q.scenario_payload(s) for s in scenarios]
    for key in ("model", "profiles", "placement", "workload", "weights_resident"):
        assert left[key] == right[key], key
    assert left["weights_resident"] is False
    assert left["hardware"]["components"] == right["hardware"]["components"]
    hbf = scenarios[0].hardware.get_component("hbf0")
    assert not hbf.is_active_memory
    assert "hbf_media" in hbf.metadata
    assert len(scenarios[0].workload.requests) == batch
    assert scenarios[0].workload.scheduler.max_num_seqs == batch
    assert all(r.prompt_tokens == 8 and r.output_tokens == 4 for r in scenarios[0].workload.requests)
    options = scenarios[0].placement.metadata["control_plane"]["policy"]["options"]
    assert "hbf0" in options["weight_tensor_targets"].values()
    assert options["kv_cache_target"] == "hbm0"
    assert options["linear_state_target"] == "hbm2"

    selected = {"gpu-hbf0", "gpu-hbm0", "gpu-hbm1", "gpu-hbm2"}
    shared_owners, independent_owners = set(), set()
    links_before = {link["link_id"]: link for link in left["hardware"]["links"]}
    links_after = {link["link_id"]: link for link in right["hardware"]["links"]}
    assert links_before.keys() == links_after.keys()
    for link_id in links_before:
        a, b = deepcopy(links_before[link_id]), deepcopy(links_after[link_id])
        if link_id in selected:
            owners = []
            for payload, link in ((left, a), (right, b)):
                owner = link["metadata"].pop("physical_resource_owner", "link." + link_id)
                owner = payload["hardware"]["metadata"].get("physical_resource_owners", {}).get("link." + link_id, owner)
                owners.append(owner)
            independent_owners.add(owners[0])
            shared_owners.add(owners[1])
        assert a == b, link_id
    assert len(independent_owners) == 4
    assert len(shared_owners) == 1
    paths = matrix.differences(left, right)
    assert paths
    assert all(matrix.allowed_change(path, group["kind"]) for path in paths), paths
    for path in ("/hardware/links/gpu-hbm7/metadata/physical_resource_owner",
                 "/hardware/metadata/physical_resource_owners/link.gpu-hbm7",
                 "/hardware/links/gpu-hbf0/bandwidth_gbps",
                 "/hardware/links/gpu-hbf0/latency_ns", "/weights_resident"):
        assert not matrix.allowed_change(path, group["kind"]), path



def test_actual_mapping_whitelist_allows_kv_but_not_other_placement_or_identity():
    observation = {"model_sha256": "model", "workload_sha256": "workload", "placement": {
        "tensor_to_component": {"kv_cache": "hbm0", "linear_state": "hbm2", "weight": "hbm1"},
        "op_to_component": {"op": "gpu0"}, "tensor_bytes": {"weight": 64},
        "weight_tensor_details": {"weight": {"total_physical_bytes": 64}},
        "rank_weight_shards": {"weight": [{"storage_component_id": "hbm1", "physical_bytes": 64}]},
        "memory_tiers": {"kv_cache_component": "hbm0", "kv_layer_components": {"layer": "hbm0"},
                         "linear_state_layer_components": {"linear": "hbm2"}},
    }}
    moved = deepcopy(observation)
    moved["placement"]["tensor_to_component"]["kv_cache"] = "hbf0"
    moved["placement"]["memory_tiers"]["kv_cache_component"] = "hbf0"
    moved["placement"]["memory_tiers"]["kv_layer_components"]["layer"] = "hbf0"
    cells = [{"observation": observation}, {"observation": moved}]
    assert matrix.actual_mapping_matches(cells, "placement")
    assert not matrix.actual_mapping_matches(cells, "topology")
    assert not matrix.actual_mapping_matches(cells, "flash_topology")
    for path, value in [
        (("model_sha256",), "other"), (("workload_sha256",), "other"),
        (("placement", "tensor_to_component", "linear_state"), "hbf0"),
        (("placement", "tensor_to_component", "weight"), "hbf0"),
        (("placement", "op_to_component", "op"), "cim0"),
        (("placement", "tensor_bytes", "weight"), 128),
        (("placement", "weight_tensor_details", "weight", "total_physical_bytes"), 128),
        (("placement", "rank_weight_shards", "weight", 0, "storage_component_id"), "hbf0"),
        (("placement", "memory_tiers", "linear_state_layer_components", "linear"), "hbf0"),
    ]:
        corrupt = deepcopy(moved)
        target = corrupt
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert not matrix.actual_mapping_matches([cells[0], {"observation": corrupt}], "placement"), path
