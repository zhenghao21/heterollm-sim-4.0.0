"""Physical matrix partitioning preserves shared inputs and distinct outputs."""
from dataclasses import replace

import pytest

from heterollm_sim import planner
from test_blackwell_physical_projection_dispatch import _mixed_case, _projection
from test_mmvf_source_dispatch import _case as _f16_case, _qkv


@pytest.mark.parametrize("family", ("mixed", "mmvf"))
@pytest.mark.parametrize("explicit_ids", (False, True))
def test_segmented_projection_addresses_and_l2_share_input_only(family, explicit_ids):
    if family == "mixed":
        scenario = _mixed_case()
        layer, workload, metadata = _projection(scenario, mixed=True, m=16)
    else:
        scenario = _f16_case()
        workload, metadata, _ = _qkv(scenario)
        layer = planner._execution_layers(scenario)[0]
    profiles = {kind: dict(items) for kind, items in scenario.component_profiles.items()}
    for name, gpu in profiles["gpu"].items():
        profiles["gpu"][name] = replace(gpu, kernel_model=replace(gpu.kernel_model, stateful_l2=True))
    scenario = replace(scenario, component_profiles=profiles)
    if explicit_ids:
        # Lower-priority tensor IDs used to get lost when the parent acquired
        # three op_names, or caused all outputs to alias the same first byte.
        metadata = {**metadata, "input_tensor_id": "shared-input", "output_tensor_id": "qkv-output"}
    with planner._compilation_scope(scenario):
        plan = planner._parallel_plan(scenario)
        builder = planner._TaskBuilder(scenario.workload.requests[0])
        planner._add_rank_gemm(
            builder, scenario, planner._topology_router(scenario), plan, plan.ranks[0],
            workload, "gpu0", "decode." + layer.layer_id + ".qkv", (),
            weight_tensor_id=layer.layer_id + ".attention_weights",
            keep_output_on_target=True, metadata=metadata,
        )
    matrices = [task for task in builder.tasks if task.metadata.get("phase") == "gpu_gemm"]
    launches = [task for task in builder.tasks if task.metadata.get("phase") == "kernel_launch"]
    assert len(matrices) == len(launches) == 3
    activations, outputs, weights = [], [], []
    for index, task in enumerate(matrices):
        accesses = task.metadata["memory_accesses"]
        activation, weight = [item for item in accesses if item["operation"] == "read"]
        output, = [item for item in accesses if item["operation"] == "write"]
        activations.append(activation)
        weights.append(weight)
        outputs.append(output)
        assert activation["byte_count"] == workload.activation_bytes
        cache_input, = [item for item in task.metadata["stateful_l2"]["accesses"]
                        if item["buffer_id"] == activation["buffer_id"]]
        assert cache_input["size_bytes"] == activation["byte_count"]
        if explicit_ids:
            assert activation["buffer_id"].startswith("tensor:shared-input:")
            assert output["buffer_id"].startswith("tensor:qkv-output:segment:" + str(index) + ":")
        if index:
            assert matrices[index - 1].task_id in launches[index].dependencies
    assert len({item["buffer_id"] for item in activations}) == 1
    assert len({item["address"] for item in activations}) == 1
    assert len({item["buffer_id"] for item in weights}) == 3
    assert len({item["buffer_id"] for item in outputs}) == 3
    assert len({item["address"] for item in outputs}) == 3
    assert sum(item["byte_count"] for item in weights) == workload.weight_bytes
    assert sum(item["byte_count"] for item in outputs) == workload.output_bytes


@pytest.mark.parametrize("key", ("input_buffer_id", "input_tensor_id", "activation_tensor_id", "input_allocation_id", "allocation_id"))
def test_segment_input_identity_preserves_every_supported_explicit_key(key):
    metadata = planner._physical_projection_buffer_metadata({key: "input"}, "projection", 1)
    assert metadata["input_buffer_id"] == "input"


@pytest.mark.parametrize("key", ("output_buffer_id", "output_tensor_id", "output_tensor", "output_allocation_id", "allocation_id"))
def test_segment_output_identity_partitions_every_supported_explicit_key(key):
    metadata = planner._physical_projection_buffer_metadata({key: "output"}, "projection", 1)
    assert metadata["output_buffer_id"] == "output:segment:1"


@pytest.mark.parametrize("metadata", ({"write_alias_of": "other"}, {"weight_alias_of": "other"},
    {"alias_of": "other"}, {"output_offset_bytes": 128}, {"weight_offset_bytes": 128}))
def test_partition_refuses_unrepresented_parent_views(metadata):
    with pytest.raises(ValueError, match="independent weight/output storage"):
        planner._physical_projection_buffer_metadata(metadata, "projection", 1)
