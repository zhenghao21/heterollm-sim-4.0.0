"""Operation-engine semantics, using explicitly synthetic reduced model geometry.

These tests do not represent paper-scale performance or detailed OCP reference
runs. They exercise private memory, ownership, actual I/O and replay invariants.
"""
from dataclasses import replace
from math import ceil
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from exploring_hbf_paper_config import hardware_by_id
from heterollm_sim.paper_inference import (
    PaperExecutionAssumptions, PaperInferenceEngine, PaperNetwork, PaperParallelism,
)
from heterollm_sim.paper_memory import PaperMemoryCapacityError
from heterollm_sim.paper_models import LLAMA3_405B, LLAMA4_MAVERICK
from heterollm_sim.paper_serving_control import ServingRequest


def synthetic_model(*, moe=False, layers=8, chunk=0):
    """Named synthetic dimensions prevent a small fixture posing as a 405B run."""
    return replace(LLAMA4_MAVERICK if moe else LLAMA3_405B,
        model_id="synthetic_test_model", name="Synthetic reduced semantic test geometry",
        layers=layers, hidden_size=64, intermediate_size=128, query_heads=8,
        kv_heads=2, head_dim=8, vocabulary_size=128, max_context_tokens=1024,
        expert_intermediate_size=64 if moe else 0, routed_experts=4 if moe else 0,
        top_k=1 if moe else 0, shared_experts=1 if moe else 0,
        moe_layer_step=2 if moe else 0, attention_chunk_size=chunk,
        full_attention_interval=4 if chunk else 1,
        official_total_parameters_rounded=0, official_active_parameters_rounded=0,
        checkpoint_geometry_verified=False, geometry_verification="synthetic_unit_test",
        paper_reported_weight_capacity_gb=None)


def engine(*, hardware="HBF+", parallel=None, batch=2, model=None, compressed=False,
           input_tokens=10, output_tokens=5):
    return PaperInferenceEngine(model or synthetic_model(), hardware_by_id(hardware),
        parallel or PaperParallelism(), batch_size=batch, input_tokens=input_tokens,
        output_tokens=output_tokens,
        assumptions=PaperExecutionAssumptions(use_period_compression=compressed))


def requests(batch, *, input_tokens=10, output_tokens=5, ages=None):
    return tuple(ServingRequest(index, input_tokens, output_tokens,
        0 if ages is None else ages[index], slot_id=index) for index in range(batch))


def imported_start(execution, batch):
    return max(execution.import_kv(request, 0.).ready_ns for request in batch)


def assert_close_nested(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for name in actual:
            assert_close_nested(actual[name], expected[name])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_close_nested(left, right)
    elif isinstance(actual, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-8)
    else:
        assert actual == expected


def test_replica_capacity_cannot_pool_independent_dp_gpu_memories():
    # Each independent DP copy needs the full >800 GB model; four private
    # 288 GB memories cannot substitute for a tensor-sharded replica.
    assert LLAMA3_405B.total_parameters * 2 > 288_000_000_000
    with pytest.raises(PaperMemoryCapacityError, match="private GPU"):
        engine(hardware="HBM4", model=LLAMA3_405B,
               parallel=PaperParallelism(dp=4), batch=1)
    tensor_sharded = engine(hardware="HBM4", model=LLAMA3_405B,
                            parallel=PaperParallelism(tp=4), batch=1)
    assert len(tensor_sharded.memory) == 4
    assert tensor_sharded.capacity["accepted"]
    assert all(row["device_capacity_bytes"] == 288_000_000_000
               for row in tensor_sharded.capacity["stages"])


def test_sram_workspace_capacity_is_private_per_gpu_not_pooled_by_ep():
    # 50 local logits rows * 2M vocabulary * FP32 = 400 MB per GPU,
    # exceeding 320 MB on each GPU although their combined SRAM is 640 MB.
    model = replace(synthetic_model(moe=True, layers=2), vocabulary_size=2_000_000)
    with pytest.raises(PaperMemoryCapacityError, match="private GPU"):
        engine(model=model, parallel=PaperParallelism(ep=2), batch=100)


@pytest.mark.parametrize("hardware", ["HBM4", "HBF", "HBF+"])
def test_private_gpu_resource_calendars_and_byte_counters_are_independent(hardware):
    execution = engine(hardware=hardware, parallel=PaperParallelism(tp=2))
    assert execution.memory[0] is not execution.memory[1]
    before = execution.memory[1].summary()
    tier = "hbf" if execution.hardware.hbf_stacks else "hbm"
    execution._access(0, tier, "write", 100_001, 0., "private-gpu-zero")
    assert execution.memory[1].summary() == before
    assert execution.memory[0].bytes_by_operation[tier + ":write"] >= 100_001


def test_slot_order_not_active_list_position_selects_ep_owned_kv():
    model = synthetic_model(moe=True, layers=2)
    parallel = PaperParallelism(ep=2)
    ordered = requests(4, ages=(0, 1, 2, 3))
    shuffled = (ordered[3], ordered[0], ordered[2], ordered[1])
    first = engine(model=model, parallel=parallel, batch=4)
    second = engine(model=model, parallel=parallel, batch=4)
    first_start = imported_start(first, ordered)
    second_start = imported_start(second, ordered)
    first_result = first.execute_decode(ordered, (), first_start)
    second_result = second.execute_decode(shuffled, (), second_start)
    assert first_result == second_result
    assert_close_nested([memory.summary() for memory in first.memory],
                        [memory.summary() for memory in second.memory])
    # Equal weight/endpoint reads cancel. EP 1 owns contexts 11,13 while
    # EP 0 owns 10,12: only their old-KV read amount differs.
    kv_per_token_layer = 2 * model.kv_heads * model.head_dim * 2
    expected_delta = ((11 + 13) - (10 + 12)) * model.layers * kv_per_token_layer
    assert (first.memory[1].bytes_by_operation["hbf:read"]
            - first.memory[0].bytes_by_operation["hbf:read"]) == expected_delta


def test_pd_import_uses_preselected_ep_slot_even_if_request_id_parity_differs():
    execution = engine(model=synthetic_model(moe=True, layers=2),
                       parallel=PaperParallelism(ep=2), batch=4)
    # The replacement id and its inherited slot deliberately have different parity.
    request = ServingRequest(100, 10, 5, slot_id=1)
    imported = execution.import_kv(request, 0.)
    assert imported.ready_ns > 0
    assert execution.memory[0].bytes_by_operation.get("hbf:write", 0) == 0
    assert execution.memory[1].bytes_by_operation["hbf:write"] == imported.physical_program_bytes
    assert "ib:rx:1" in execution.network.calendars
    assert "ib:rx:0" not in execution.network.calendars


def test_pd_import_and_remote_gpu_transfer_share_ib_receive_port():
    assumptions = PaperExecutionAssumptions(network_latency_ns=3.,
                                            infiniband_gb_s=10., nvlink_gb_s=100.)
    network = PaperNetwork(assumptions)
    first = network.transfer(8, 0, 100, 0.)
    second = network.receive_pd(0, 200, 0.)
    assert first == 13
    assert second == 36
    assert network.calendars["ib:rx:0"] == 36
    assert network.busy["ib:rx:0"] == 36
    assert network.bytes == 300
    assert network.operations == 2
    # Receiving on GPU 0 does not occupy the receive NIC on GPU 1.
    assert network.receive_pd(1, 100, 0.) == 13


def test_pd_receive_contention_delays_cross_node_collective_but_not_nvlink():
    assumptions = PaperExecutionAssumptions(network_latency_ns=3.,
                                            infiniband_gb_s=10., nvlink_gb_s=100.)
    isolated = PaperNetwork(assumptions)
    baseline = isolated.collective((0, 8), 100, 0.)
    shared = PaperNetwork(assumptions)
    shared.receive_pd(0, 200, 0.)
    assert shared.collective((0, 8), 100, 0.) > baseline
    nvlink = PaperNetwork(assumptions)
    nvlink.receive_pd(0, 200, 0.)
    assert nvlink.collective((0, 1), 100, 0.) == PaperNetwork(assumptions).collective((0, 1), 100, 0.)


def test_node_nic_scope_shares_pd_receive_between_all_eight_local_gpus():
    assumptions = PaperExecutionAssumptions(network_latency_ns=3., infiniband_gb_s=10.,
                                            nvlink_gb_s=100., nic_scope="node")
    node = PaperNetwork(assumptions)
    first = node.receive_pd(0, 1000, 0.)
    second = node.receive_pd(7, 100, 0.)
    independent_node = node.receive_pd(8, 100, 0.)
    assert first == 103
    assert second == 116
    assert independent_node == 13
    assert set(node.calendars) == {"ib:rx:0", "ib:rx:1"}
    assert node.transfer(8, 7, 100, 0.) == 129
    # Full-duplex node transmit and within-node NVLink have their own actors.
    assert node.transfer(0, 9, 100, 0.) == 26
    assert node.transfer(0, 7, 100, 0.) == 4
    gpu = PaperNetwork(replace(assumptions, nic_scope="gpu"))
    gpu.receive_pd(0, 1000, 0.)
    assert gpu.receive_pd(7, 100, 0.) == 13


def test_node_scope_engine_pd_targets_actually_reserve_one_shared_receive_nic():
    model = synthetic_model(moe=True, layers=2)
    execution = PaperInferenceEngine(model, hardware_by_id("HBF+"),
        PaperParallelism(ep=2, tp=2), batch_size=4, input_tokens=10, output_tokens=5,
        assumptions=PaperExecutionAssumptions(nic_scope="node", use_period_compression=False))
    execution.import_kv(ServingRequest(100, 10, 5, slot_id=0), 0.)
    execution.import_kv(ServingRequest(101, 10, 5, slot_id=1), 0.)
    assert set(execution.network.calendars) == {"ib:rx:0"}
    assert execution.network.operations == 4
    assert all(memory.bytes_by_operation.get("hbf:write", 0) > 0 for memory in execution.memory)


def test_layer_trace_obeys_attention_ffn_and_expert_dispatch_dependencies():
    execution = engine(model=synthetic_model(moe=True, layers=2),
                       parallel=PaperParallelism(ep=2, tp=2), batch=4)
    batch = requests(4)
    start = imported_start(execution, batch)
    for memory in execution.memory:
        memory.retain_trace = True
    collectives = []
    original = execution.network.collective

    def observed(ranks, size, ready_ns, **options):
        end = original(ranks, size, ready_ns, **options)
        collectives.append((tuple(ranks), options.get("kind", "all_reduce"), ready_ns, end))
        return end

    execution.network.collective = observed
    # One isolated real MoE layer keeps repeated labels unambiguous.
    execution._layer(0, 1, [request.context_tokens for request in batch], start)
    dispatches = [call for call in collectives if call[1] == "all_to_all"]
    assert len(dispatches) == 4  # Two tensor groups: outbound dispatch and return combine.
    first_dispatch = min(call[2] for call in dispatches)
    for memory in execution.memory:
        events = memory.events

        def one_compute(label):
            matches = [event for event in events if event.operation == "compute" and event.label == label]
            assert len(matches) == 1
            return matches[0]

        qkv = one_compute("qkv-projection")
        attention = one_compute("attention-kv")
        softmax = one_compute("softmax")
        output = one_compute("attention-output-projection")
        assert attention.ready_ns >= qkv.end_ns
        assert softmax.ready_ns >= attention.end_ns
        assert output.ready_ns >= softmax.end_ns
        router = one_compute("expert-router")
        selection = one_compute("router-select")
        logits = [event for event in events if event.label == "router-logits"]
        top_k = [event for event in events if event.label == "router-top-k"]
        assert logits and top_k
        assert min(event.ready_ns for event in logits) >= router.end_ns
        assert min(event.ready_ns for event in top_k) >= max(event.end_ns for event in logits)
        assert selection.ready_ns >= max(event.end_ns for event in top_k)
        assert first_dispatch >= selection.end_ns
        for path in ("shared", "routed"):
            gate = one_compute(path + "-gate-up")
            writes = [event for event in events if event.label == path + "-gate-up-output"]
            reads = [event for event in events if event.label == path + "-swiglu-input"]
            activation = one_compute(path + "-swiglu")
            down = one_compute(path + "-down")
            assert writes and reads
            assert min(event.ready_ns for event in writes) >= gate.end_ns
            assert min(event.ready_ns for event in reads) >= max(event.end_ns for event in writes)
            assert activation.ready_ns >= max(event.end_ns for event in reads)
            assert down.ready_ns >= activation.end_ns
        assert one_compute("routed-gate-up").ready_ns >= one_compute("shared-down").end_ns


def test_embedding_tp_all_gather_finishes_before_first_attention_norm():
    execution = engine(model=synthetic_model(moe=True, layers=2),
                       parallel=PaperParallelism(ep=2, tp=2), batch=4)
    batch = requests(4)
    start = imported_start(execution, batch)
    for memory in execution.memory:
        memory.retain_trace = True
    gathered = []
    original = execution.network.collective

    def observed(ranks, size, ready_ns, **options):
        end = original(ranks, size, ready_ns, **options)
        if options.get("kind") == "all_gather":
            gathered.append((tuple(ranks), ready_ns, end))
        return end

    execution.network.collective = observed
    execution.execute_decode(batch, (), start)
    assert {ranks for ranks, _, _ in gathered} == {(0, 1), (2, 3)}
    for ranks, ready, end in gathered:
        assert end > ready
        for rank in ranks:
            events = execution.memory[rank].events
            embedding = next(event for event in events
                             if event.operation == "compute" and event.label == "embedding")
            norm = next(event for event in events
                        if event.operation == "compute" and event.label == "attention-norm-weight")
            assert ready >= embedding.end_ns
            assert norm.ready_ns >= end


def test_independent_ep_groups_do_not_serialize_lm_head_tensor_collectives():
    execution = engine(model=synthetic_model(moe=True, layers=2),
                       parallel=PaperParallelism(ep=2, tp=2), batch=4)
    batch = requests(4)
    start = imported_start(execution, batch)
    calls = []
    collective = execution.network.collective

    def observed(ranks, size, ready_ns, **options):
        end = collective(ranks, size, ready_ns, **options)
        calls.append((tuple(ranks), size, ready_ns, end))
        return end

    execution.network.collective = observed
    execution.execute_decode(batch, (), start)
    first, second = calls[-2:]
    assert first[0] == (0, 1)
    assert second[0] == (2, 3)
    assert first[1] == second[1] == 32
    # Their GPU/memory/NVLink actors are private and symmetric. The sampling
    # reduction in one EP group cannot become another EP group's dependency.
    assert second[2] == pytest.approx(first[2], rel=1e-12, abs=1e-8)
    assert second[3] == pytest.approx(first[3], rel=1e-12, abs=1e-8)


@pytest.mark.parametrize("hardware", ["HBF", "HBF+", "CONV", "CONV+"])
def test_pd_and_decode_physical_program_bytes_equal_actual_padded_memory_writes(hardware):
    execution = engine(hardware=hardware, model=synthetic_model(layers=2), batch=2)
    for memory in execution.memory:
        memory.retain_trace = True
    batch = requests(2)
    imports = [execution.import_kv(request, 0.) for request in batch]
    pd_events = [event for memory in execution.memory for event in memory.events
                 if event.tier == "hbf" and event.operation == "write" and event.label == "pd-import"]
    assert sum(item.physical_program_bytes for item in imports) == sum(event.size_bytes for event in pd_events)
    assert all(event.size_bytes % 4096 == 0 for event in pd_events)
    assert all(item.physical_program_bytes > item.logical_kv_bytes for item in imports)
    before = sum(memory.bytes_by_operation.get("hbf:write", 0) for memory in execution.memory)
    decoded = execution.execute_decode(batch, (), max(item.ready_ns for item in imports))
    decode_events = [event for memory in execution.memory for event in memory.events
                     if event.tier == "hbf" and event.operation == "write" and event.label == "decode-kv"]
    actual_delta = sum(memory.bytes_by_operation.get("hbf:write", 0) for memory in execution.memory) - before
    assert decoded.physical_program_bytes == actual_delta == sum(event.size_bytes for event in decode_events)
    assert decoded.logical_write_bytes == 2 * execution.model.layers * 2 * 2 * 8 * 2
    assert decoded.physical_program_bytes > decoded.logical_write_bytes
    assert all(event.size_bytes == 4096 for event in decode_events)
    # Packed page interleaving does not fan a sub-page write out to every
    # stack and manufacture multiple physical pages for one small group.
    assert len(decode_events) == execution.model.layers


@pytest.mark.parametrize("hardware", ["HBM4", "HBF", "HBF+", "CONV", "CONV+"])
@pytest.mark.parametrize("moe,parallel,chunk", [
    (False, PaperParallelism(), 0),
    (True, PaperParallelism(ep=2, tp=2), 16),
])
def test_period_replay_matches_uncompressed_time_bytes_counters_and_calendars(hardware, moe, parallel, chunk):
    model = synthetic_model(moe=moe, layers=16 if chunk else 8, chunk=chunk)
    slow = engine(hardware=hardware, model=model, parallel=parallel, batch=4, compressed=False, input_tokens=14)
    fast = engine(hardware=hardware, model=model, parallel=parallel, batch=4, compressed=True, input_tokens=14)
    batch = requests(4, input_tokens=14, ages=(0, 1, 2, 3))
    slow_start = imported_start(slow, batch)
    fast_start = imported_start(fast, batch)
    first = slow.execute_decode(batch, (), slow_start)
    second = fast.execute_decode(tuple(reversed(batch)), (), fast_start)
    assert first.end_ns == pytest.approx(second.end_ns, rel=1e-12, abs=1e-7)
    assert first.physical_program_bytes == second.physical_program_bytes
    assert first.logical_write_bytes == second.logical_write_bytes
    assert_close_nested(first.costs, second.costs)
    assert fast.layer_execution_count == slow.layer_execution_count == model.layers
    assert fast.compressed_periods > 0
    assert slow.compressed_periods == 0
    for a, b in zip(slow.memory, fast.memory):
        assert_close_nested(a.summary(), b.summary())
    assert slow.network.bytes == fast.network.bytes
    assert slow.network.operations == fast.network.operations
    assert_close_nested(slow.network.calendars, fast.network.calendars)
    assert_close_nested(slow.network.busy, fast.network.busy)


def test_small_closed_loop_measures_replacements_and_separates_dp_scaling():
    execution = engine(model=synthetic_model(layers=2), batch=2,
                       parallel=PaperParallelism(dp=4))
    result = execution.run(warmup_steps=5, measurement_steps=10)
    assert result["completed_requests"] >= 4
    assert result["full_batch_step_fraction"] == 1
    assert result["pd_program_completed_bytes"] > 0
    assert result["cluster_decode_tokens_per_second"] == 4 * result["decode_tokens_per_second"]
    assert result["per_gpu_decode_tokens_per_second"] == result["decode_tokens_per_second"]
    assert len(result["gpu_memories"]) == 1
    assert result["old_single_request_fast_mode_used"] is False
    assert not result["step_records"]
