"""Small arithmetic and communication contracts; no hardware calibration."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.batched_accelerator import BatchedGemmInput, evaluate_batched_gemm
from heterollm_sim.communication import RouteHop, TopologyRouter, plan_collective
from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.conversion_work import (
    ConversionSourceContract, SOURCE_SHA256, UnsupportedConversion, derive_conversion_work,
)
from heterollm_sim.cost_models import (
    CacheHierarchyProfile, CacheLevelProfile, CPUPipelineProfile, CPUProfile,
    FusedAttentionWorkload, GemmWorkload, GPUProfile, HBMProfile,
    HostMemoryProfile, TensorCoreProfile, TensorKernelWorkload, DigitalSramCimProfile,
    estimate_cpu_gemm, estimate_gpu_fused_attention, estimate_gpu_gemm,
    estimate_gpu_tensor_kernel, estimate_cim_gemm,
)
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.ir import ComponentSpec, HardwareSpec, LayerSpec, LinkSpec
from heterollm_sim.parallel import _layer_stage_cost, shard_extent
from heterollm_sim.precision import canonical_dtype, layer_precision_bits


def cache():
    return CacheHierarchyProfile((CacheLevelProfile("L", 1 << 20, 64, 0, 10000),))


def gpu(**changes):
    values = dict(tensor_core=TensorCoreProfile(1, 1, 1,
        dtype_throughput_scale={"bf16": .5}), cache_hierarchy=cache(),
        scalar_lanes_per_sm=32, scalar_ops_per_cycle=1,
        reduction_ops_per_cycle_per_sm=1, special_function_units_per_sm=1,
        special_function_ops_per_cycle=1)
    values.update(changes)
    return GPUProfile(**values)


@pytest.mark.parametrize("attention", [False, True])
def test_explicit_bf16_selects_its_own_compute_throughput(attention):
    device, memory = gpu(), HBMProfile(1000)
    workload = (FusedAttentionWorkload(16, 16, 16, activation_dtype="fp16")
        if attention else GemmWorkload(16, 16, 16,
            activation_bits=16, weight_bits=16, activation_dtype="fp16"))
    estimate = estimate_gpu_fused_attention if attention else estimate_gpu_gemm
    fp16 = estimate(device, memory, workload)
    bf16 = estimate(device, memory, replace(workload, activation_dtype="bf16"))
    key = "tensor_service_ns" if attention else "compute_service_ns"
    assert fp16.metadata["tensor_dtype"] == "fp16"
    assert bf16.metadata["tensor_dtype"] == "bf16"
    assert bf16.metadata[key] == fp16.metadata[key] * 2
    assert bf16.bytes_moved == fp16.bytes_moved
    # A valid BF16-only device must not require unrelated FP16 support.
    only_bf16 = replace(device, default_tensor_dtype="bf16",
        tensor_core=replace(device.tensor_core, supported_dtypes=("bf16",)))
    assert estimate(only_bf16, memory, replace(workload, activation_dtype="bf16"))


@pytest.mark.parametrize("head_dim", [16, 24, 32])
def test_attention_mma_padding_is_per_independent_head(head_dim):
    device = gpu()
    work = FusedAttentionWorkload(16, 16, 8 * head_dim, score_heads=8,
                                  kv_hidden_size=2 * head_dim)
    result = estimate_gpu_fused_attention(device, HBMProfile(1000), work)
    per_product = 8 * ((head_dim + 15) // 16)
    assert result.metadata["qk_tile_count"] == per_product
    assert result.metadata["pv_tile_count"] == per_product
    issued = 2 * per_product * device.tensor_core.operations_per_mma
    assert result.metadata["issued_tensor_operations"] == issued
    assert result.metadata["tensor_operations"] == 4 * 16 * 16 * 8 * head_dim
    demand = next(d for p in result.phases for d in p.demands
                  if d.resource_id == device.tensor_core.resource_id)
    assert result.metadata["kv_read_bytes"] == 2 * 16 * 2 * head_dim * 2


def test_attention_rejects_fractional_head_width():
    with pytest.raises(ValueError, match="divisible"):
        FusedAttentionWorkload(1, 16, 65, score_heads=8)


def test_pure_sfu_kernel_has_service_and_work_without_fake_memory():
    device = gpu()
    work = TensorKernelWorkload(0, 0, 0, transcendental_operations=10)
    result = estimate_gpu_tensor_kernel(device, HBMProfile(1000), work)
    assert result.service_ns == 10.
    assert result.bytes_moved == 0
    sfu = next(d for p in result.phases for d in p.demands
               if d.resource_id == device.special_function_resource_id)
    assert sfu.work_units == 10


def test_cpu_gops_gpu_tops_and_max_service_units():
    work = GemmWorkload(16, 16, 16, activation_bits=16, weight_bits=16)
    device = gpu( kernel_launch_ns=5.)
    result = estimate_gpu_gemm(device, HBMProfile(1000), work)
    assert result.metadata["compute_service_ns"] == 1.
    assert result.metadata["memory_service_ns"] == 1536 / 1000
    assert result.service_ns == 5. + max(1., 1536 / 1000)
    cpu = CPUProfile(CPUPipelineProfile(1, 1, 16, 4, 4, 4), cache())
    host = estimate_cpu_gemm(cpu, HostMemoryProfile(1000), work)
    assert host.metadata["throughput_compute_service_ns"] == 8192 / 2


@pytest.mark.parametrize("m", [1, 3])
def test_cpu_generic_quantized_transform_matches_scheduled_work(m):
    cpu = CPUProfile(CPUPipelineProfile(1, 1, 16, 4, 4, 4), cache(),
        )
    work = GemmWorkload(m, 32, 4, activation_bits=16, weight_bits=4,
        packed_weight_formats=("Q4_0",), packed_weight_transform_operations=100,
        packed_weight_format_segments=(("Q4_0", 4, 100),))
    result = estimate_cpu_gemm(cpu, HostMemoryProfile(1000), work)
    operations = 100 * m
    schedule = result.metadata["instruction_schedule"]
    assert schedule["packed_weight_transform_instructions"] == operations


def test_batch_evaluator_units_overlap_and_stable_ranking():
    result = evaluate_batched_gemm(BatchedGemmInput(
        memory_bandwidth_model="nominal_bandwidth",
        m=[16, 16], k=16, n=16, activation_bits=16, weight_bits=16,
        peak_tops=8.192, memory_bandwidth_gb_s=1000,
        kernel_launch_ns=5, io_bytes=80, io_bandwidth_gb_s=8, io_latency_ns=3,
        communication_bytes=80, communication_bandwidth_gbps=8,
        communication_latency_ns=7, candidate_ids=("first", "second")), backend="numpy")
    assert list(result.compute_ns) == [1., 1.]
    assert list(result.memory_ns) == [1.536, 1.536]
    assert list(result.io_ns) == [13., 13.]
    assert list(result.communication_ns) == [87., 87.]
    assert list(result.total_ns) == [106.536, 106.536]
    assert result.ranked_candidate_ids == ("first", "second")


def test_batch_evaluator_does_not_silently_discard_complex_input():
    with pytest.raises(ValueError, match="实数"):
        evaluate_batched_gemm(BatchedGemmInput(m=16 + 1j, k=16, n=16,
            memory_bandwidth_model="nominal_bandwidth",
            peak_tops=1, memory_bandwidth_gb_s=1), backend="numpy")


def router(*, shared=True, depth=1):
    nodes = tuple(ComponentSpec(c, "gpu") for c in ("a", "b"))
    edge = LinkSpec("ab", "a", "p", "b", "p", "pcie", bandwidth_gbps=8,
        latency_ns=20, metadata={"shared_bidirectional": shared,
            "queue_depth": depth})
    return TopologyRouter(HardwareSpec("test", nodes, (edge,)))


@pytest.mark.parametrize("shared, expected", [(True, 220.), (False, 120.)])
def test_link_direction_sharing_serializes_bytes_not_latency_twice(shared, expected):
    fabric = router(shared=shared, depth=2)
    tasks = []
    for index, (source, target) in enumerate((("a", "b"), ("b", "a"))):
        phase, = fabric.transfer_phases(source, target, 100)
        tasks.append(TaskSpec(str(index), "request", "transfer", TaskCategory.COMMUNICATION,
                              demands=phase.demands, metadata=phase.metadata))
    kernel = UnifiedEventKernel(resource_capacities=fabric.resource_capacities)
    kernel.submit(tasks)
    rows = [kernel.step(), kernel.step()]
    assert max(row.end_ns for row in rows) == expected
    assert sum(d.bytes_moved for task in tasks for d in task.demands) == 200


def test_zero_byte_hop_has_no_startup():
    hop = RouteHop("link", "a", "b", "wire", 8, 50)
    assert hop.transfer_ns(0) == 0
    assert hop.demands(0) == ()


def test_routing_compares_latency_and_size_using_declared_bit_bandwidth():
    nodes = tuple(ComponentSpec(c, "gpu") for c in ("a", "b", "c"))
    links = tuple(LinkSpec(name, source, "p", target, "p", "pcie",
        bandwidth_gbps=bw, latency_ns=latency)
        for name, source, target, bw, latency in (
            ("slow_direct", "a", "b", 8, 0),
            ("fast_first", "a", "c", 800, 20),
            ("fast_second", "c", "b", 800, 20)))
    fabric = TopologyRouter(HardwareSpec("routes", nodes, links))
    assert [hop.link_id for hop in fabric.route("a", "b", 1)] == ["slow_direct"]
    route = fabric.route("a", "b", 1000)
    assert [hop.link_id for hop in route] == ["fast_first", "fast_second"]
    assert sum(hop.transfer_ns(1000) for hop in route) == 60


@pytest.mark.parametrize("kind, factor", [
    ("all_reduce", 2), ("all_gather", 1), ("reduce_scatter", 1), ("all_to_all", 1),
])
def test_ring_collective_exact_aggregate_traffic_and_all_to_all_destinations(kind, factor):
    participants = ("a", "b", "c")
    fabric = SimpleNamespace(components=dict.fromkeys(participants), route=lambda *args, **kw: ())
    plan = plan_collective(fabric, kind, participants, 12288, algorithm="ring")
    assert len(plan.rounds) == factor * 2
    assert sum(t.byte_count for r in plan.rounds for t in r.transfers) == factor * 2 * 12288
    if kind == "all_to_all":
        assert {(t.source_component, t.target_component) for r in plan.rounds for t in r.transfers} == {
            (source, target) for source in participants for target in participants if source != target}


@pytest.mark.parametrize("algorithm", ["ring", "tree"])
def test_collective_padding_keeps_large_integer_payload_exact(algorithm):
    participants = ("a", "b")
    fabric = SimpleNamespace(components=dict.fromkeys(participants), route=lambda *args, **kw: ())
    size = 2**53 + 1
    plan = plan_collective(fabric, "reduce_scatter", participants, size, algorithm=algorithm)
    chunk = (size + 1) // 2
    assert plan.rounds[-1].transfers[0].byte_count == chunk


@pytest.mark.parametrize("head_dim, gated", [(8, False), (16, True), (16, False)])
def test_pp_balancing_respects_declared_attention_and_ffn_geometry(head_dim, gated):
    layer = LayerSpec("layer", "dense", 64, 128, 8, kv_heads=2,
                      attention_head_dim=head_dim, gated_mlp=gated)
    q = 8 * head_dim
    attention = 64 * (q + 2 * 2 * head_dim) + q * 64
    ffn = (3 if gated else 2) * 64 * 128
    assert _layer_stage_cost(layer) == attention + ffn


def test_sharding_never_loses_elements_or_accepts_fractional_dimensions():
    assert shard_extent(2**53 + 1, 1, 0).padding == 0
    for degree in (1, 3, 8):
        extent = shard_extent(13, degree, degree - 1)
        assert extent.local_size * degree >= 13
        assert extent.padding == extent.local_size * degree - 13
    for args in ((True, 1, 0), (16., 2, 0), (16, 2.5, 0), (16, 2, .5)):
        with pytest.raises(ValueError):
            shard_extent(*args)


def test_precision_storage_contract():
    assert canonical_dtype("BFloat16") == "bf16"
    assert layer_precision_bits("bf16", "W4A8", unsupported_dtype_message="dtype") == (8, 4)


@pytest.mark.parametrize("path, weight_format", [
    ("MMVQ_Q8_1", "Q5_0"), ("MMQ_D4", "Q8_0"), ("MMQ_DS4", "Q4_1"),
])
def test_conversion_source_dag_preserves_input_padding_output_and_dependencies(path, weight_format):
    contract = ConversionSourceContract(1200, 1200, 32, SOURCE_SHA256, "0" * 64, True)
    work = derive_conversion_work(m=3, k=544, weight_format=weight_format, path=path, contract=contract)
    assert work.k_padded == 1024
    assert work.read_bytes == 3 * 544 * 4
    assert work.write_bytes == 3 * 1024 * 36 // 32
    assert work.launched_threads * work.values_per_thread == work.padded_elements
    seen = set()
    for node in work.dag:
        assert set(node.dependencies) <= seen
        assert node.node_id not in seen
        seen.add(node.node_id)
    stores = sum(node.count for node in work.dag if node.operation.startswith("store_"))
    assert stores == work.write_bytes
    assert work.to_metadata()["cost_model_applied"] is False
    with pytest.raises(UnsupportedConversion, match="aligned"):
        derive_conversion_work(m=3, k=543, weight_format=weight_format, path=path, contract=contract)


def test_cim_bit_slices_reduction_and_cold_load_conserve_work():
    profile = DigitalSramCimProfile(array_count=4, p_m=1, p_k=4, p_n=4,


         )
    work = GemmWorkload(2, 8, 8, activation_bits=2, weight_bits=2)
    cold = estimate_cim_gemm(profile, work)
    warm = estimate_cim_gemm(profile, work, weights_resident=True)
    assert cold.metadata["array_cycles"] == 8
    array = cold.phase("cim_array_eval")
    assert next(d.work_units for d in array.demands if d.resource_id == profile.array_resource_id) == 32
    assert cold.phase("cim_accumulate_reduce").metadata["accumulation_ops"] == 112
    assert cold.service_ns - warm.service_ns == cold.phase("weight_load").service_ns


@pytest.mark.parametrize("changes", [
    {"epilogue_operations": 8}, {"epilogue_transcendental_operations": 8},
])
def test_cim_rejects_unmodeled_epilogues_instead_of_discarding_work(changes):
    work = GemmWorkload(1, 16, 16, epilogue_name="activation", **changes)
    for resident in (False, True):
        with pytest.raises(ValueError, match="explicit peripheral operator"):
            estimate_cim_gemm(DigitalSramCimProfile(), work, weights_resident=resident)
