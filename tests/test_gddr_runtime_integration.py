"""Small end-to-end checks for the real GPU/GDDR lowering path."""

from dataclasses import replace
import json
from threading import Thread
from urllib.request import Request, urlopen

import pytest

from heterollm_sim.component_presets import get_component_preset
from heterollm_sim.communication import TopologyRouter
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.cost_models import CostPhase, GDDRProfile
from heterollm_sim.data_motion import PhysicalRuntimeContext, endpoint_service, resolve_physical_task
from heterollm_sim.dram_core import DramCore
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.gguf_parity import GGUFTensor, GGUFMetadata, build_model_from_gguf
from heterollm_sim.ir import (
    LayerSpec,
    LinearAttentionSpec,
    RankMappingSpec,
    RequestSpec,
    build_model_graph_from_layer_specs,
)
from heterollm_sim.kernel_model import KernelCapability, KernelModelProfile
from heterollm_sim.memory_types import AccessRequest, Operation, parse_physical_memory_config
from heterollm_sim.planner import (
    _direct_memory_address,
    _direct_memory_phase,
    _gddr_stable_address,
    _gddr_allocation_size,
    _linear_state_endpoint_binding,
    _promote_physical_allocation_extents,
    _weight_source_is_compute_local_backing,
    compile_scenario,
    compile_serving_cohort_schedule,
    validate_scenario,
)
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serving import BatchCohort, BatchItem
from heterollm_sim.web import build_server, scenario_to_payload


def _gpu_gddr_scenario():
    """Replace one reference HBM stack with a connected GDDR7 device."""

    reference = build_reference_scenario()
    gpu = reference.hardware.get_component("gpu0")
    old_memory = reference.hardware.get_component("hbm0")
    gddr = get_component_preset("gddr7-16gb-30_0-256bit").component
    metadata = dict(gddr.metadata)
    # The preset id is intentionally independent of this test's component id;
    # let ComponentSpec/config bind the physical owner to hbm0 at construction.
    metadata.pop("memory_service_owner", None)
    memory = replace(
        gddr,
        component_id="hbm0",
        package_id=old_memory.package_id,
        die_id=old_memory.die_id,
        metadata=metadata,
    )
    gpu = replace(
        gpu,
        ports=tuple(
            replace(
                port,
                protocol="GDDR",
                version="GDDR7",
                lanes=256,
                bandwidth_gbps=7680.0,
            )
            if port.port_id == "hbm0"
            else port
            for port in gpu.ports
        ),
        metadata={"supported_gddr_generations": ["GDDR7"]},
    )
    links = tuple(
        replace(
            link,
            protocol="GDDR",
            version="GDDR7",
            lanes=256,
            bandwidth_gbps=7680.0,
        )
        if link.link_id == "gpu-hbm0"
        else link
        for link in reference.hardware.links
    )
    components = tuple(
        gpu if component.component_id == "gpu0" else
        memory if component.component_id == "hbm0" else component
        for component in reference.hardware.components
    )
    profiles = {kind: dict(registry) for kind, registry in reference.component_profiles.items()}
    profiles["gddr"] = {
        "legacy-gddr": GDDRProfile(
            bandwidth_gb_s=960.0,
            efficiency=0.75,
            energy_pj_per_byte=4.0,
            resource_id="gddr7_16gb_30_0_256bit.memory",
            read_latency_ns=35.0,
            write_latency_ns=35.0,
            transaction_bytes=256,
            max_outstanding_requests=64,
            parallel_lanes=8,
            generation="GDDR7",
        )
    }
    return replace(
        reference,
        hardware=replace(
            reference.hardware,
            components=components,
            links=links,
        ),
        component_profiles=profiles,
    )


def _post_json(url, payload):
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def test_real_gpu_gddr_compile_carries_gemm_attention_and_directional_physical_accesses():
    scenario = _gpu_gddr_scenario()
    assert validate_scenario(scenario).is_valid
    schedule = compile_scenario(scenario)

    physical = [
        task
        for task in schedule.tasks
        if task.metadata.get("physical_memory_config")
        and task.metadata.get("memory_accesses")
    ]
    assert physical, "the real reference workload must lower GPU work to GDDR descriptors"
    assert any(task.metadata.get("phase") == "gpu_gemm" for task in physical)
    assert any("attention_flash.gpu_fused_attention" in task.name for task in physical)

    gemm = next(task for task in physical if task.name.endswith("qkv.gpu_gemm"))
    cost = gemm.metadata["cost_model"]
    assert cost["activation_bytes"] + cost["weight_bytes"] > 0
    assert cost["output_bytes"] > 0
    accesses = tuple(gemm.metadata["memory_accesses"])
    assert sum(item["byte_count"] for item in accesses if item["operation"] == "read") > 0
    assert sum(item["byte_count"] for item in accesses if item["operation"] == "write") == cost["output_bytes"]
    capacity = gemm.metadata["physical_memory_config"]["capacity_bytes"]
    assert all(
        0 <= item["address"] < capacity
        and item["address"] + item["byte_count"] <= capacity
        for item in accesses
    )


def test_direct_memory_address_prefers_metadata_and_aligns_stable_fallback():
    scenario = _gpu_gddr_scenario()
    rank = scenario.placement.parallel.rank_mapping[0]
    component = scenario.hardware.get_component("hbm0")
    capacity = component.metadata["physical_memory_config"]["capacity_bytes"]
    burst = component.metadata["physical_memory_config"]["burst_bytes"]
    metadata = {"layer_id": "layer-007", "operator_id": "qkv"}

    first = _direct_memory_address(scenario, rank, "hbm0", 1024, metadata)
    second = _direct_memory_address(scenario, rank, "hbm0", 1024, metadata)
    assert first == second
    assert first % burst == 0
    assert 0 <= first <= capacity - 1024
    assert _direct_memory_address(
        scenario,
        rank,
        "hbm0",
        1024,
        {**metadata, "page_offset_bytes": 4096},
    ) == 4096


def test_direct_memory_phase_drops_expanded_physical_trace_from_phase_metadata():
    """A planner preview must keep counters, not one tuple per DRAM burst."""

    scenario = _gpu_gddr_scenario()
    placement = replace(
        scenario.placement,
        metadata={
            "llama_backend_memory": {
                "hbm0": {"access": "direct", "device_id": "gpu0"},
            },
        },
    )
    scenario = replace(scenario, placement=placement)
    # Use HBM1 as the logical local backing so hbm0 is exercised through the
    # direct-addressed GDDR path.
    rank = replace(
        scenario.placement.parallel.rank_mapping[0],
        memory_component_id="hbm1",
    )
    phase = CostPhase(
        "gpu_gemm",
        TaskCategory.COMPUTE,
        (ResourceDemand("hbm1.hbm_fabric", 10.0, bytes_moved=1024),),
        metadata={"read_bytes": 1024, "write_bytes": 0},
    )

    lowered = _direct_memory_phase(
        scenario,
        rank,
        phase,
        "hbm0",
        read_bytes=6 * 1024 * 1024,
    )
    direct = lowered.metadata["direct_memory_access"]
    bill = direct["read_service"]
    assert bill["physical_bytes"] > 0
    assert bill["details_truncated"] is True
    assert "resource_intervals" not in bill
    assert "resource_interval_payloads" not in bill
    assert "physical_resource_intervals" not in bill


def test_planner_endpoint_preview_can_drop_expanded_physical_trace():
    scenario = _gpu_gddr_scenario()
    component = scenario.hardware.get_component("hbm0")
    full = endpoint_service(
        component,
        1024 * 1024,
        read=True,
        name="full-preview",
        page_offset_bytes=0,
    )
    compact = endpoint_service(
        component,
        1024 * 1024,
        read=True,
        name="compact-preview",
        page_offset_bytes=0,
        compact_preview=True,
    )
    assert full is not None and compact is not None
    full_bill = full.metadata["physical_execution"]
    compact_bill = compact.metadata["physical_execution"]
    assert full_bill["physical_bytes"] == compact_bill["physical_bytes"]
    assert full_bill["resource_intervals"]
    assert compact_bill["details_truncated"] is True
    assert "resource_intervals" not in compact_bill
    assert "resource_interval_payloads" not in compact_bill


def test_communication_preview_preserves_timing_and_dispatch_trace():
    component = _gpu_gddr_scenario().hardware.get_component("hbm0")
    phase = TopologyRouter._endpoint_phase(component, 65536, read=True,
                                          name="communication", page_offset_bytes=0)
    full = endpoint_service(component, 65536, read=True, name="full", page_offset_bytes=0)
    assert phase.demands[0].service_ns == pytest.approx(full.demands[0].service_ns)
    assert phase.demands[0].bytes_moved == full.demands[0].bytes_moved
    assert "resource_intervals" not in phase.metadata["physical_execution"]
    task = TaskSpec(task_id="endpoint", request_id="request", name=phase.name,
                    category=TaskCategory.COMMUNICATION, demands=phase.demands, metadata=phase.metadata)
    dispatched = resolve_physical_task(task, PhysicalRuntimeContext(capture_details=True), 0)
    assert dispatched.metadata["physical_execution"]["resource_intervals"]


def test_promoted_gddr_extent_rehashes_inferred_stable_address():
    scenario = _gpu_gddr_scenario()
    config = scenario.hardware.get_component("hbm0").metadata["physical_memory_config"]
    buffer_id = "tensor:final_norm.output:rank=0:tp_rank=0:pp_rank=0"
    first_address = _gddr_stable_address(
        buffer_id, 0, 2, config["capacity_bytes"], config["burst_bytes"], 2
    )
    second_address = _gddr_stable_address(
        buffer_id, 0, 2048, config["capacity_bytes"], config["burst_bytes"], 2048
    )
    def task(task_id, byte_count, address):
        return TaskSpec(
            task_id=task_id,
            request_id="cohort-000000",
            name=task_id,
            category=TaskCategory.COMPUTE,
            demands=(ResourceDemand("hbm0.gddr_fabric", 1.0, bytes_moved=byte_count),),
            metadata={
                "physical_memory_config": config,
                "memory_accesses": ({
                    "operation": "write", "address": address,
                    "byte_count": byte_count, "offset_bytes": 0,
                    "allocation_generation": 1, "generation": 1,
                    "buffer_id": buffer_id,
                    "address_source": "stable_buffer_tensor_offset",
                },),
            },
        )

    promoted = _promote_physical_allocation_extents((
        task("norm.reduce", 2, first_address),
        task("norm.apply", 2048, second_address),
    ))
    addresses = [row["address"] for item in promoted for row in item.metadata["memory_accesses"]]
    extents = [row["allocation_size_bytes"] for item in promoted for row in item.metadata["memory_accesses"]]
    assert addresses == [second_address, second_address]
    assert extents == [2048, 2048]


def test_gddr_weight_projection_does_not_claim_fused_tensor_extent():
    """Each MLP projection must allocate its own shard, not the fused group."""

    scenario = _gpu_gddr_scenario()
    placement = replace(
        scenario.placement,
        tensor_bytes={"layer-000.mlp_weights": 149_422_080},
    )
    scenario = replace(scenario, placement=placement)
    metadata = {
        "projection_id": "mlp.up_gate",
        "weight_tensor_id": "layer-000.mlp_weights",
    }
    assert _gddr_allocation_size(
        scenario,
        metadata,
        "weight",
        "tensor:layer-000.mlp_weights:projection_id=mlp.up_gate",
        0,
        99_614_720,
    ) is None


def test_aggregate_physical_runtime_keeps_scalar_timing_without_stage_trace():
    """Aggregate cohort execution must not retain burst-sized stage tuples."""

    scenario = _gpu_gddr_scenario()
    raw_config = scenario.hardware.get_component("hbm0").metadata[
        "physical_memory_config"
    ]
    config = parse_physical_memory_config(raw_config)
    core = DramCore(config, capture_details=False)
    result = core.submit(
        AccessRequest(
            "aggregate-preview",
            Operation.READ,
            0,
            6 * 1024 * 1024,
            0.0,
        )
    )
    assert result.completion_ns > 0
    assert result.transfer_bytes > 0
    assert result.mapping == ()
    assert result.stages == ()
    assert result.counters["burst_count"] > 0


def test_closed_physical_schedule_releases_transient_generation_buffers():
    """Sequential serving activations must not accumulate in the GDDR allocator."""

    raw = dict(_gpu_gddr_scenario().hardware.get_component("hbm0").metadata[
        "physical_memory_config"
    ])
    raw["capacity_bytes"] = 64 * 1024 * 1024
    owner = "hbm0.memory"
    size = 48 * 1024 * 1024

    def task(task_id, buffer_id, dependencies=()):
        return TaskSpec(
            task_id=task_id,
            request_id="cohort-000000",
            name=task_id,
            category=TaskCategory.MEMORY,
            dependencies=tuple(dependencies),
            demands=(ResourceDemand(owner, 1.0, bytes_moved=size),),
            metadata={
                "physical_memory_config": raw,
                "physical_owner": owner,
                "memory_accesses": ({
                    "operation": "write",
                    "buffer_id": buffer_id,
                    "offset_bytes": 0,
                    "byte_count": size,
                    "allocation_size_bytes": size,
                    "allocation_generation": 1,
                    "generation": 1,
                    "address_source": "stable_buffer_tensor_offset",
                    "physical_owner": owner,
                    "resource_id": owner,
                },),
            },
        )

    schedule = (task("activation.0", "activation.0"),
                task("activation.1", "activation.1", ("activation.0",)))
    kernel = UnifiedEventKernel.from_closed_graph(
        schedule,
        resource_capacities={owner: 1},
    )
    while kernel.has_active_tasks:
        assert kernel.step() is not None
    assert kernel.physical_runtime.allocators[owner].allocations() == ()


def test_frontend_transport_round_trip_accepts_default_gddr_component(tmp_path):
    """Exercise the same normalize/validate transport used by the Web UI."""

    scenario = _gpu_gddr_scenario()
    payload = scenario_to_payload(scenario)
    # The UI saves JSON, then reloads it and asks /normalize for canonical
    # memory-service metadata.  Keep this test at the actual HTTP boundary.
    server = build_server("127.0.0.1", 0, component_preset_cache_dir=tmp_path)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = "http://127.0.0.1:{}".format(server.server_address[1])
        normalized = _post_json(base_url + "/api/normalize", payload)
        reloaded = scenario_from_dict(normalized)
        component = reloaded.hardware.get_component("hbm0")
        physical = component.metadata["physical_memory_config"]
        assert component.kind == "gddr"
        assert physical["kind"] == "GDDR"
        assert physical["generation"] == "GDDR7"
        assert component.capacity_bytes == physical["capacity_bytes"]
        assert component.capacity_bytes == 16_000_000_000
        validation = _post_json(base_url + "/api/validate", normalized)
        assert validation["valid"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _tiny_dense_gddr_scenario(prompt_tokens, *, stateful_l2):
    """Keep the reference topology, replacing its workload with tiny real ops."""

    scenario = _gpu_gddr_scenario()
    graph = build_model_graph_from_layer_specs(
        "tiny-dense-transformer",
        (
            LayerSpec(
                layer_id="tiny0",
                kind="dense",
                hidden_size=8,
                intermediate_size=16,
                attention_heads=2,
                kv_heads=1,
                dtype="int8",
                quantization="w8a8",
                weight_bytes=512,
            ),
        ),
        architecture="decoder_only_transformer",
        vocabulary_size=16,
        max_sequence_length=16,
        embedding_weight_bytes=128,
        output_weight_bytes=128,
    )
    model = replace(scenario.model, name="tiny-dense-transformer", graph=graph)

    gpu_profile = scenario.component_profiles["gpu"]["legacy-gpu"]
    # The tiny cache deliberately fits two lines.  This makes natural reuse,
    # eviction, and dirty writeback observable without synthetic cache probes.
    levels = list(gpu_profile.cache_hierarchy.levels)
    levels[0] = replace(
        levels[0], capacity_bytes=128, associativity=1, banks=1,
        read_ports=1, write_ports=1, max_outstanding=2,
    )
    levels[-1] = replace(
        levels[-1], capacity_bytes=256, associativity=1, banks=1,
        read_ports=1, write_ports=1, max_outstanding=2,
    )
    tiny_gpu = replace(
        gpu_profile,
        cache_hierarchy=replace(gpu_profile.cache_hierarchy, levels=tuple(levels)),
        kernel_model=KernelModelProfile(
            hardware_id="tiny-gddr-test",
            runtime_id="analytical",
            architecture="tiny-dense-transformer",
            stateful_l2=stateful_l2,
            kernels=(
                *(
                    KernelCapability(
                        kernel_family="tiny-{}-{}-{}".format(
                            phase, output_bits, epilogue or "plain"
                        ),
                        weight_formats=("int8",),
                        activation_dtype="int8",
                        phase=phase,
                        compute_primitive="simt",
                        internal_dtype="int8",
                        evidence="explicit analytical tiny test kernel contract",
                        output_bits=output_bits,
                        epilogue=epilogue,
                        cta_geometry=(1, 8, 8),
                    )
                    for phase, output_bits, epilogue in (
                        ("prefill", 8, ""),
                        ("prefill", 16, ""),
                        ("prefill", 16, "rope"),
                        ("prefill", 16, "swiglu"),
                        ("decode", 8, ""),
                        ("decode", 16, ""),
                        ("decode", 16, "rope"),
                        ("decode", 16, "swiglu"),
                    )
                ),
                *(
                    KernelCapability(
                        kernel_family="tiny-attention-{}".format(phase),
                        weight_formats=("int8",),
                        activation_dtype="int8",
                        phase=phase,
                        compute_primitive="simt",
                        internal_dtype="int8",
                        operator="attention",
                        evidence="explicit analytical tiny test attention contract",
                        output_bits=8,
                        cta_geometry=(1, 8, 8),
                    )
                    for phase in ("prefill", "decode")
                ),
                *(
                    KernelCapability(
                        kernel_family="tiny-fp16-{}-{}".format(
                            phase, epilogue or "plain"
                        ),
                        weight_formats=("fp16",),
                        activation_dtype="fp16",
                        phase=phase,
                        compute_primitive="simt",
                        internal_dtype="fp16",
                        evidence="explicit analytical imported-F16 test kernel contract",
                        output_bits=16,
                        epilogue=epilogue,
                        cta_geometry=(1, 8, 8),
                    )
                    for phase in ("prefill", "decode")
                    for epilogue in ("", "rope", "swiglu")
                ),
                *(
                    KernelCapability(
                        kernel_family="tiny-int8-f32-{}".format(phase),
                        weight_formats=("int8",),
                        activation_dtype="int8",
                        phase=phase,
                        compute_primitive="simt",
                        internal_dtype="int8",
                        evidence="explicit analytical F32-hidden test kernel contract",
                        output_bits=32,
                        epilogue=epilogue,
                        cta_geometry=(1, 8, 8),
                    )
                    for phase in ("prefill", "decode")
                    for epilogue in ("", "swiglu")
                ),
                *(
                    KernelCapability(
                        kernel_family="tiny-fp16-int8kv-attention-{}".format(phase),
                        weight_formats=("int8",),
                        activation_dtype="fp16",
                        phase=phase,
                        compute_primitive="simt",
                        internal_dtype="fp16",
                        operator="attention",
                        evidence="explicit analytical imported-F16 INT8-KV test contract",
                        output_bits=16,
                        cta_geometry=(1, 8, 8),
                    )
                    for phase in ("prefill", "decode")
                ),
            ),
        ),
    )
    profiles = {kind: dict(registry)
                for kind, registry in scenario.component_profiles.items()}
    profiles["gpu"]["legacy-gpu"] = tiny_gpu

    placement = replace(
        scenario.placement,
        model_name=model.name,
        parallel=replace(
            scenario.placement.parallel,
            layer_to_stage={"tiny0": 0},
        ),
        kv_policy=replace(
            scenario.placement.kv_policy,
            tokens_per_page=4,
            dtype="int8",
            offload_ratio=0.0,
        ),
    )
    workload = replace(
        scenario.workload,
        name="tiny-dense-workload",
        requests=(RequestSpec(
            request_id="tiny-request",
            arrival_ns=0.0,
            prompt_tokens=prompt_tokens,
            output_tokens=3,
        ),),
        request_count=1,
        prompt_tokens=prompt_tokens,
        output_tokens=3,
        mtp=None,
        scheduler=replace(
            scenario.workload.scheduler,
            max_num_seqs=1,
            max_num_batched_tokens=16,
            prefill_chunk_tokens=prompt_tokens,
        ),
    )
    return replace(
        scenario,
        name="tiny-dense-gpu-gddr",
        model=model,
        placement=placement,
        workload=workload,
        component_profiles=profiles,
    )


def test_imported_qwen3_weights_use_inferred_gpu_memory_without_duplicate_transfer():
    scenario = _tiny_dense_gddr_scenario(2, stateful_l2=False)
    tensors = []
    offset = 0
    for name, shape in (
        ("blk.0.attn_q.weight", (8, 8)),
        ("blk.0.attn_k.weight", (8, 4)),
        ("blk.0.attn_v.weight", (8, 4)),
        ("blk.0.attn_output.weight", (8, 8)),
        ("blk.0.ffn_gate.weight", (8, 16)),
        ("blk.0.ffn_up.weight", (8, 16)),
        ("blk.0.ffn_down.weight", (16, 8)),
        ("token_embd.weight", (8, 16)),
        ("output.weight", (8, 16)),
    ):
        size = shape[0] * shape[1] * 2
        tensors.append(GGUFTensor(name, shape, 1, "F16", 1, size, offset))
        offset += size
    model = build_model_from_gguf(GGUFMetadata(
        "tiny-qwen3.gguf", "test-directory", 3, len(tensors), 0,
        "qwen3", 1, 8, 2, 1, 16, 16, "MOSTLY_F16", 1, {}, tuple(tensors),
    ))
    scenario = replace(
        scenario,
        model=model,
        placement=replace(
            scenario.placement,
            model_name=model.name,
            tensor_to_component={"model_weights": "hbm0"},
            tensor_bytes={"model_weights": offset},
            parallel=replace(
                scenario.placement.parallel,
                rank_mapping=(),
                layer_to_stage={},
            ),
        ),
    )
    schedule = compile_scenario(scenario)
    accesses = [task for task in schedule.tasks
                if task.metadata.get("event_kind") == "model_weight_access"]
    assert accesses
    assert all(task.metadata["weight_source_is_compute_local_backing"]
               for task in accesses)
    assert all(task.metadata["weight_source_transfer_emitted"] is False
               for task in accesses)
    assert not any(task.metadata.get("event_kind") == "model_weight_read"
                   for task in schedule.tasks)
    physical = [task for task in schedule.tasks
                if task.metadata.get("physical_memory_config")]
    assert physical, "inferred local weights must retain real GDDR accesses"
    assert any(access["operation"] == "read" and "weights" in access.get("buffer_id", "")
               for task in physical for access in task.metadata["memory_accesses"])
    kernel = UnifiedEventKernel.from_closed_graph(
        schedule.tasks,
        resource_capacities=schedule.resource_capacities,
        resource_owners=schedule.resource_owners,
    )
    while kernel.has_active_tasks:
        assert kernel.step() is not None


def test_gpu_local_weight_backing_preserves_explicit_rank_memory_selection():
    scenario = _gpu_gddr_scenario()
    assert _weight_source_is_compute_local_backing(scenario, "hbm0", "gpu0")
    rank = replace(scenario.placement.parallel.rank_mapping[0], memory_component_id="hbm1")
    scenario = replace(
        scenario,
        placement=replace(scenario.placement, parallel=replace(
            scenario.placement.parallel, rank_mapping=(rank,),
        )),
    )
    assert _weight_source_is_compute_local_backing(scenario, "hbm1", "gpu0")
    assert not _weight_source_is_compute_local_backing(scenario, "hbm0", "gpu0")


def test_gpu_weight_backing_predicate_keeps_unavailable_memory_non_local():
    scenario = _gpu_gddr_scenario()
    scenario = replace(
        scenario,
        hardware=replace(
            scenario.hardware,
            components=(scenario.hardware.get_component("gpu0"),
                        scenario.hardware.get_component("cpu0")),
            links=(),
        ),
        placement=replace(scenario.placement, parallel=replace(
            scenario.placement.parallel, rank_mapping=(),
        )),
    )
    assert not _weight_source_is_compute_local_backing(scenario, "external_memory", "gpu0")


def _tiny_linear_gddr_scenario(*, direct=False):
    scenario = _tiny_dense_gddr_scenario(2, stateful_l2=False)
    graph = build_model_graph_from_layer_specs(
        "tiny-hybrid",
        tuple(LayerSpec(
            layer_id="linear-{:03d}".format(index), kind="dense", hidden_size=8,
            intermediate_size=16, attention_heads=2, kv_heads=1,
            sequence_mixer="linear_attention",
            linear_attention=LinearAttentionSpec(
                key_heads=1, value_heads=2, key_head_dim=2, value_head_dim=2,
                conv_kernel_size=3,
            ),
            dtype="int8", quantization="w8a8", weight_bytes=512,
            metadata={"gguf_tensor_bindings": [{"offset": 10**12}]},
        ) for index in range(2)),
        vocabulary_size=16, max_sequence_length=16,
        embedding_weight_bytes=128, output_weight_bytes=128,
    )
    model = replace(scenario.model, name="tiny-hybrid", graph=graph)
    metadata = dict(scenario.placement.metadata)
    if direct:
        metadata["llama_backend_memory"] = {"hbm0": {"access": "direct", "device_id": "gpu0"}}
    return replace(scenario, model=model, placement=replace(
        scenario.placement, model_name=model.name,
        tensor_to_component={"linear_state": "hbm0"}, tensor_bytes={"linear_state": 192},
        parallel=replace(scenario.placement.parallel, layer_to_stage={}), metadata=metadata,
    ))


@pytest.mark.parametrize("direct", (False, True))
def test_linear_state_reads_and_writes_share_allocations_without_aliasing_other_layers(direct):
    scenario = _tiny_linear_gddr_scenario(direct=direct)
    schedule = compile_scenario(scenario)
    state_tasks = [task for task in schedule.tasks
                   if task.metadata.get("event_kind") in {"linear_state_read", "linear_state_write"}
                   and task.metadata.get("physical_memory_config")]
    assert state_tasks
    previews = {}
    for task in state_tasks:
        access = task.metadata["memory_access"]
        assert access["buffer_id"].startswith("linear_state:")
        assert access["address_source"] == "stable_buffer_tensor_offset"
        assert access["generation"] == 0
        assert access["address"] + access["allocation_size_bytes"] <= task.metadata["physical_memory_config"]["capacity_bytes"]
        previews.setdefault(access["buffer_id"], set()).add(access["address"])
    assert len(previews) == 2
    assert all(len(addresses) == 1 for addresses in previews.values())
    kernel = UnifiedEventKernel.from_closed_graph(
        schedule.tasks, resource_capacities=schedule.resource_capacities,
        resource_owners=schedule.resource_owners,
    )
    ranges = {}
    operations = {}
    while kernel.has_active_tasks:
        event = kernel.step()
        assert event is not None
        if (event.task.metadata.get("event_kind") not in {"linear_state_read", "linear_state_write"}
                or not event.task.metadata.get("physical_memory_config")):
            continue
        for access in event.task.metadata["memory_accesses"]:
            ranges.setdefault(access["buffer_id"], set()).add((access["address"], access["allocation_size_bytes"]))
            operations.setdefault(access["buffer_id"], set()).add(access["operation"])
    assert all(values == {"read", "write"} for values in operations.values())
    assert all(len(values) == 1 for values in ranges.values())
    (first, size), (second, other_size) = [next(iter(values)) for values in ranges.values()]
    assert first + size <= second or second + other_size <= first


def test_direct_linear_state_shared_memory_preserves_each_rank_read_write_identity():
    scenario = _tiny_linear_gddr_scenario(direct=True)
    scenario = replace(scenario, placement=replace(
        scenario.placement, parallel=replace(
            scenario.placement.parallel, tp_degree=2,
            rank_mapping=tuple(RankMappingSpec(
                rank=rank, component_id="gpu0", tp_rank=rank, pp_rank=0, ep_rank=0,
                memory_component_id="hbm0",
            ) for rank in range(2)),
        ),
    ))
    schedule = compile_scenario(scenario)
    state_tasks = [task for task in schedule.tasks
                   if task.metadata.get("event_kind") in {"linear_state_read", "linear_state_write"}
                   and task.metadata.get("physical_memory_config")]
    preview_ids = {}
    for task in state_tasks:
        assert task.metadata["direct_memory_component"] == "hbm0"
        owner = (task.metadata["layer_id"], task.metadata["rank"])
        access = task.metadata["memory_access"]
        assert "rank={}:".format(owner[1]) in access["buffer_id"]
        preview_ids.setdefault(owner, set()).add(access["buffer_id"])
    assert set(preview_ids) == {(layer, rank) for layer in ("linear-000", "linear-001")
                               for rank in range(2)}
    assert all(len(identities) == 1 for identities in preview_ids.values())
    assert len(set.union(*preview_ids.values())) == 4
    state_task_ids = {task.task_id for task in state_tasks}
    kernel = UnifiedEventKernel.from_closed_graph(
        schedule.tasks, resource_capacities=schedule.resource_capacities,
        resource_owners=schedule.resource_owners,
    )
    ranges = {}
    operations = {}
    while kernel.has_active_tasks:
        event = kernel.step()
        assert event is not None
        if event.task.task_id not in state_task_ids:
            continue
        for access in event.task.metadata["memory_accesses"]:
            ranges.setdefault(access["buffer_id"], set()).add(
                (access["address"], access["allocation_size_bytes"])
            )
            operations.setdefault(access["buffer_id"], set()).add(access["operation"])
    assert len(ranges) == 4
    assert all(values == {"read", "write"} for values in operations.values())
    assert all(len(values) == 1 for values in ranges.values())
    ordered = sorted(next(iter(values)) for values in ranges.values())
    assert all(size == 48 for _, size in ordered)
    assert all(base + size <= next_base
               for (base, size), (next_base, _) in zip(ordered, ordered[1:]))


def test_linear_state_endpoint_preserves_authored_address_and_rejects_capacity_overflow():
    scenario = _tiny_linear_gddr_scenario()
    metadata = {"layer_id": "linear-000", "rank": 0, "state_storage": "persistent_state"}
    binding = _linear_state_endpoint_binding(scenario, "hbm0", 96, metadata, "cohort-0", 4096)
    assert binding["address"] == 4096
    assert binding["address_source"] == "explicit_physical_address"
    capacity = scenario.hardware.get_component("hbm0").metadata["physical_memory_config"]["capacity_bytes"]
    with pytest.raises(ValueError, match="exceeds physical capacity"):
        _linear_state_endpoint_binding(scenario, "hbm0", 96, metadata, "cohort-0", capacity - 1)


def test_serving_linear_state_uses_actual_request_owner_across_synthetic_cohorts():
    scenario = _tiny_linear_gddr_scenario()
    identities = []
    for index, request_id in enumerate(("request-a", "request-a", "request-b")):
        cohort = BatchCohort(
            cohort_id="cohort-{:06d}".format(index), kind="decode", start_ns=0.0,
            items=(BatchItem(request_id, "decode", 1, 2 + index, logit_tokens=1),),
        )
        schedule = compile_serving_cohort_schedule(scenario, cohort)
        accesses = [task.metadata["memory_access"] for task in schedule.tasks
                    if task.metadata.get("event_kind") in {"linear_state_read", "linear_state_write"}
                    and task.metadata.get("physical_memory_config")]
        assert accesses
        assert all(access["generation"] == 0 for access in accesses)
        assert all("owner=" + request_id in access["buffer_id"] for access in accesses)
        identities.append({(access["buffer_id"], access["address"]) for access in accesses})
    assert identities[0] == identities[1]
    assert identities[0].isdisjoint(identities[2])


def test_linear_state_multi_owner_transfer_keeps_separate_allocations_and_total_bytes():
    scenario = _tiny_linear_gddr_scenario()
    binding = _linear_state_endpoint_binding(
        scenario, "hbm0", 192,
        {"layer_id": "linear-000", "rank": 0, "linear_state_independent_owner_count": 2},
        "cohort-000000", owner_request_ids=("request-a", "request-b"),
    )
    accesses = binding["accesses"]
    assert len(accesses) == 2
    assert len({access["buffer_id"] for access in accesses}) == 2
    assert sum(access["byte_count"] for access in accesses) == 192
    assert all(access["allocation_size_bytes"] == 96 for access in accesses)


def test_linear_state_speculative_buffer_is_distinct_and_scoped_to_cohort():
    scenario = _tiny_linear_gddr_scenario()
    metadata = {"layer_id": "linear-000", "rank": 0, "state_storage": "rolling_speculative_buffer"}
    first = _linear_state_endpoint_binding(scenario, "hbm0", 96, metadata, "cohort-000000",
                                          owner_request_ids=("request-a",))
    repeated = _linear_state_endpoint_binding(scenario, "hbm0", 96, metadata, "cohort-000000",
                                             owner_request_ids=("request-a",))
    next_cohort = _linear_state_endpoint_binding(scenario, "hbm0", 96, metadata, "cohort-000001",
                                                owner_request_ids=("request-a",))
    persistent = _linear_state_endpoint_binding(scenario, "hbm0", 96,
                                                {**metadata, "state_storage": "persistent_state"},
                                                "cohort-000000", owner_request_ids=("request-a",))
    assert first == repeated
    assert first["buffer_id"] == next_cohort["buffer_id"]
    assert first["generation"] > 0 and first["generation"] != next_cohort["generation"]
    assert first["buffer_id"] != persistent["buffer_id"]
    assert persistent["generation"] == 0


def test_recurrent_scan_keeps_work_linear_without_materializing_per_token_states():
    scenario = _tiny_linear_gddr_scenario()
    scenario = replace(scenario, workload=replace(
        scenario.workload,
        metadata={**scenario.workload.metadata, "llama_cpp_f32_hidden_storage": True},
        scheduler=replace(scenario.workload.scheduler, max_num_batched_tokens=512,
                          max_num_ubatch_tokens=512, prefill_chunk_tokens=512),
    ))
    scans_by_tokens = {}
    state_extents = {}
    for tokens in (1, 512):
        cohort = BatchCohort(
            cohort_id="cohort-000000", kind="prefill", start_ns=0.0,
            items=(BatchItem("request-a", "prefill", tokens, tokens, logit_tokens=1),),
        )
        schedule = compile_serving_cohort_schedule(scenario, cohort)
        scans = [task for task in schedule.tasks
                 if task.metadata.get("linear_op") == "scan_recurrent_update"
                 and task.metadata.get("physical_memory_config")]
        assert len(scans) == 2
        scans_by_tokens[tokens] = scans
        state_extents[tokens] = {
            access["buffer_id"]: access["allocation_size_bytes"]
            for task in schedule.tasks
            if task.metadata.get("event_kind") in {"linear_state_read", "linear_state_write"}
            and task.metadata.get("physical_memory_config")
            for access in task.metadata.get("memory_accesses", (task.metadata["memory_access"],))
        }
        assert len(state_extents[tokens]) == 2
        assert set(state_extents[tokens].values()) == {96}
        reductions = {task.metadata["layer_id"]: task for task in schedule.tasks
                      if task.metadata.get("linear_op") == "gate_norm_reduce"
                      and task.metadata.get("physical_memory_config")}
        for task in scans:
            cost = task.metadata["cost_model"]
            # Tiny geometry: Q/K/V widths 2/2/4; value output is T*4 F32.
            assert cost["read_bytes"] == tokens * 8 * 4
            assert cost["write_bytes"] == tokens * 4 * 4
            assert task.metadata["output_bytes"] == tokens * 4 * 4
            assert task.metadata["state_io_accounting"] == "separate_linear_state_tasks"
            output_access = next(access for access in task.metadata["memory_accesses"]
                                 if access["operation"] == "write")
            assert output_access["allocation_size_bytes"] == tokens * 4 * 4
            reduction = reductions[task.metadata["layer_id"]]
            input_access = next(access for access in reduction.metadata["memory_accesses"]
                                if access["operation"] == "read")
            assert input_access["buffer_id"] == output_access["buffer_id"]
            assert input_access["byte_count"] == output_access["byte_count"]
        kernel = UnifiedEventKernel.from_closed_graph(
            schedule.tasks, resource_capacities=schedule.resource_capacities,
            resource_owners=schedule.resource_owners,
        )
        while kernel.has_active_tasks:
            assert kernel.step() is not None
        allocator = kernel.physical_runtime.allocators["hbm0.memory"]
        for buffer_id, extent in state_extents[tokens].items():
            allocation = allocator.lookup(buffer_id, 0)
            assert allocation.size_bytes == extent
    assert state_extents[1] == state_extents[512]
    for short, long in zip(scans_by_tokens[1], scans_by_tokens[512]):
        assert long.metadata["analytical_ops"] == 512 * short.metadata["analytical_ops"]


@pytest.mark.parametrize("prompt_tokens", (2, 5))
@pytest.mark.parametrize("stateful_l2", (False, True))
def test_tiny_dense_gddr_runs_prefill_and_two_decode_steps(
    prompt_tokens, stateful_l2,
):
    scenario = _tiny_dense_gddr_scenario(
        prompt_tokens, stateful_l2=stateful_l2,
    )
    payload = scenario_to_payload(scenario)
    reloaded = scenario_from_dict(payload)
    assert validate_scenario(reloaded).is_valid
    schedule = compile_scenario(reloaded)
    physical_tasks = [
        task for task in schedule.tasks
        if task.metadata.get("physical_memory_config")
    ]
    assert physical_tasks
    assert any(task.metadata.get("target_component") == "gpu0"
               and task.metadata["physical_memory_config"]["kind"] == "GDDR"
               for task in physical_tasks)
    assert any(task.metadata["physical_memory_config"]["kind"] == "DDR"
               for task in physical_tasks)

    kernel = UnifiedEventKernel.from_closed_graph(
        schedule.tasks,
        resource_capacities=schedule.resource_capacities,
        resource_owners=schedule.resource_owners,
    )
    events = []
    while kernel.has_active_tasks:
        event = kernel.step()
        assert event is not None
        events.append(event)
    assert not kernel.has_active_tasks
    phases = {event.task.metadata.get("phase") for event in events}
    assert "prefill" in phases
    assert {"decode0001", "decode0002"}.issubset(phases)

    physical_events = [
        event for event in events
        if "physical_execution" in event.task.metadata
    ]
    assert physical_events
    assert any(event.task.metadata["physical_memory_config"]["kind"] == "DDR"
               for event in physical_events)
    gddr_events = [event for event in physical_events
                   if event.task.metadata["physical_memory_config"]["kind"] == "GDDR"]
    assert gddr_events
    assert any(access["operation"] == "read" for event in gddr_events
               for access in event.task.metadata["memory_accesses"])
    for event in gddr_events:
        metadata = event.task.metadata
        config = metadata["physical_memory_config"]
        assert config["kind"] == "GDDR"
        assert config["generation"] == "GDDR7"
        capacity = config["capacity_bytes"]
        accesses = metadata["memory_accesses"]
        assert accesses
        for access in accesses:
            assert access["physical_owner"] == "hbm0.memory"
            assert access["resource_id"] == "hbm0.memory"
            assert access["operation"] in {"read", "write"}
            assert access["offset_bytes"] >= 0
            assert isinstance(access["generation"], int)
            assert access["generation"] >= 0
            assert 0 <= access["address"]
            assert access["address"] + access["byte_count"] <= capacity
        assert event.task.dependencies
        assert 0 <= event.dependency_ready_ns <= event.start_ns

    if stateful_l2:
        l2 = [event.task.metadata["l2_execution"]
              for event in gddr_events if "l2_execution" in event.task.metadata]
        assert any(item["miss_lines"] for item in l2)
        assert any(item["hit_lines"] for item in l2)
        assert any(item["dirty_eviction_bytes"] for item in l2)
        assert any(
            any(access["operation"] == "write"
                for access in event.task.metadata["memory_accesses"])
            and
            event.task.metadata["physical_execution"]["physical_write_bytes"] > 0
            for event in physical_events
        )
    else:
        assert all("l2_execution" not in event.task.metadata
                   for event in physical_events)


@pytest.mark.parametrize("cohort_index", (0, 1))
def test_serving_gddr_promotes_final_norm_extent_before_l2_registration(cohort_index):
    scenario = _tiny_dense_gddr_scenario(5, stateful_l2=True)
    cohort = BatchCohort(
        cohort_id="cohort-{:06d}".format(cohort_index),
        kind="prefill",
        start_ns=0.0,
        items=(BatchItem("tiny-request", "prefill", 5, 5, logit_tokens=1),),
    )
    schedule = compile_serving_cohort_schedule(scenario, cohort)
    norm_accesses = [
        access
        for task in schedule.tasks
        for access in task.metadata.get("stateful_l2", {}).get("accesses", ())
        if "final_norm.output" in access["buffer_id"]
    ]
    assert len(norm_accesses) >= 2
    assert len({access["size_bytes"] for access in norm_accesses}) > 1
    extent = max(access["offset_bytes"] + access["size_bytes"] for access in norm_accesses)
    assert all(access["buffer_size_bytes"] == extent for access in norm_accesses)
    assert all(access["allocation_generation"] == cohort_index + 1 for access in norm_accesses)

    kernel = UnifiedEventKernel.from_closed_graph(
        schedule.tasks,
        resource_capacities=schedule.resource_capacities,
        resource_owners=schedule.resource_owners,
    )
    while kernel.has_active_tasks:
        assert kernel.step() is not None
