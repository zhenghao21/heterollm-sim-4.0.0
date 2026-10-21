"""Whole-decode cache equivalence, using named synthetic model geometries.

These are semantic execution tests, not a paper-scale throughput experiment.
Both sides retain real PD imports, finite buffers and all resource calendars.
"""
from dataclasses import asdict, replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from exploring_hbf_paper_config import hardware_by_id
from heterollm_sim.paper_inference import PaperExecutionAssumptions, PaperInferenceEngine, PaperParallelism
from heterollm_sim.paper_models import LLAMA3_405B, LLAMA4_MAVERICK
from heterollm_sim.paper_serving_control import ServingRequest


def _model(*, moe=False, layers=8, chunk=0):
    return replace(LLAMA4_MAVERICK if moe else LLAMA3_405B,
        model_id="synthetic_decode_cache_model", name="Synthetic decode-cache semantic geometry",
        layers=layers, hidden_size=64, intermediate_size=128, query_heads=8,
        kv_heads=2, head_dim=8, vocabulary_size=128, max_context_tokens=1024,
        expert_intermediate_size=64 if moe else 0, routed_experts=4 if moe else 0,
        top_k=1 if moe else 0, shared_experts=1 if moe else 0,
        moe_layer_step=2 if moe else 0, attention_chunk_size=chunk,
        full_attention_interval=4 if chunk else 1,
        official_total_parameters_rounded=0, official_active_parameters_rounded=0,
        checkpoint_geometry_verified=False, geometry_verification="synthetic_semantic_test",
        paper_reported_weight_capacity_gb=None)


def _engine(*, cached, hardware="HBF+", parallel=None, batch=4, model=None, output=9):
    return PaperInferenceEngine(model or _model(), hardware_by_id(hardware),
        parallel or PaperParallelism(), batch_size=batch, input_tokens=10, output_tokens=output,
        assumptions=PaperExecutionAssumptions(use_period_compression=True,
                                             use_decode_transition_cache=cached))


def _requests(ages, *, output=9, identity=0):
    return tuple(ServingRequest(identity + slot, 10, output, age, slot_id=slot)
                 for slot, age in enumerate(ages))


def _same(actual, expected, path="root"):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys(), path
        for key in actual:
            _same(actual[key], expected[key], path + "." + str(key))
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected)):
            _same(a, b, path + f"[{index}]")
    elif isinstance(actual, float):
        # Affine replay only changes floating addition/subtraction order.
        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-6), path
    else:
        assert actual == expected, path


def _same_engine(actual, expected):
    assert actual.layer_execution_count == expected.layer_execution_count
    assert actual.compressed_periods == expected.compressed_periods
    assert actual.observed_kv_peak_bytes_per_gpu == expected.observed_kv_peak_bytes_per_gpu
    _same(actual.category_ns, expected.category_ns, "categories")
    _same(actual.capacity, expected.capacity, "capacity")
    assert len(actual.memory) == len(expected.memory)
    for rank, (a, b) in enumerate(zip(actual.memory, expected.memory)):
        _same(a.summary(), b.summary(), f"GPU{rank}")
        assert a._next_ticket == b._next_ticket
        assert a._next_lease == b._next_lease
        assert a._slot_releases == b._slot_releases
        _same(a._allocation_frontier_ns, b._allocation_frontier_ns)
        _same(a._slots, b._slots)
    _same(actual.network.calendars, expected.network.calendars, "network_frontiers")
    _same(actual.network.busy, expected.network.busy, "network_service")
    assert actual.network.bytes == expected.network.bytes
    assert actual.network.operations == expected.network.operations
    _same([actor.summary() for actor in actual.pd_importers],
          [actor.summary() for actor in expected.pd_importers], "HOST_DRAM_PD")


def _import_initial(engine, requests):
    pending = tuple(engine.import_kv(request, 0) for request in requests)
    assert all(item.logical_kv_bytes > 0 and item.transfer_bytes > 0 for item in pending)
    return max(item.ready_ns for item in pending)


@pytest.mark.parametrize("hardware,parallel,moe,chunk,batch", [
    ("HBM4", PaperParallelism(), False, 0, 4),
    ("HBM4", PaperParallelism(tp=2), False, 0, 4),
    ("HBF", PaperParallelism(), False, 0, 4),
    ("HBF+", PaperParallelism(), False, 0, 4),
    ("CONV+", PaperParallelism(), False, 0, 4),
    ("HBF+", PaperParallelism(ep=2), True, 8, 8),
    ("HBF+", PaperParallelism(pp=2, ep=2, microbatches=2), True, 8, 8),
])
def test_nonconsecutive_a_b_a_cache_hit_keeps_all_gpu_and_execution_observers(
        hardware, parallel, moe, chunk, batch):
    settings = dict(hardware=hardware, parallel=parallel, model=_model(moe=moe, chunk=chunk), batch=batch)
    baseline, cached = _engine(cached=False, **settings), _engine(cached=True, **settings)
    a = _requests(tuple(slot % 4 for slot in range(batch)))
    b = _requests(tuple(slot % 4 + 1 for slot in range(batch)), identity=100)
    times = [_import_initial(actor, a) for actor in (baseline, cached)]
    for query in (a, b, a):
        executions = []
        for index, actor in enumerate((baseline, cached)):
            program_before = sum(memory.bytes_by_operation.get("hbf:write", 0) for memory in actor.memory)
            categories_before = dict(actor.category_ns)
            execution = actor.execute_decode(query, (), times[index])
            program_after = sum(memory.bytes_by_operation.get("hbf:write", 0) for memory in actor.memory)
            assert execution.physical_program_bytes == program_after - program_before
            _same(execution.costs, {name: actor.category_ns[name] - value
                                    for name, value in categories_before.items()})
            executions.append(asdict(execution))
            times[index] = execution.end_ns
        _same(executions[0], executions[1], "decode_execution")
        _same_engine(cached, baseline)
    assert cached.decode_cache_hits >= 1
    assert baseline.decode_cache_hits == 0
    assert cached.compressed_periods > 0 or parallel.pp > 1 or chunk


def test_equal_aggregate_context_with_new_request_identities_reuses_the_exact_work():
    baseline, cached = _engine(cached=False), _engine(cached=True)
    a = _requests((0, 1, 2, 3))
    same_sum_new_ids = _requests((3, 2, 1, 0), identity=1000)
    times = [_import_initial(actor, a) for actor in (baseline, cached)]
    for query in (a, same_sum_new_ids):
        results = []
        for index, actor in enumerate((baseline, cached)):
            result = actor.execute_decode(query, (), times[index])
            results.append(asdict(result))
            times[index] = result.end_ns
        _same(results[0], results[1])
    assert cached.decode_cache_hits == 1
    _same_engine(cached, baseline)


@pytest.mark.parametrize("enabled", [False, True])
def test_aggregate_signature_collision_cannot_bypass_the_full_batch_contract(enabled):
    execution = _engine(cached=enabled, batch=4)
    valid = _requests((0, 1, 2, 3))  # contexts 10+11+12+13 = 46
    now = execution.execute_decode(valid, (), _import_initial(execution, valid)).end_ns
    # Two individually valid requests can share the aggregate signature of
    # four requests, but require a different amount of arithmetic/weights.
    # Full-batch validation must happen before consulting that signature.
    invalid = tuple(ServingRequest(slot, 20, 9, 3, slot_id=slot) for slot in range(2))
    before = [memory.summary() for memory in execution.memory]
    layers = execution.layer_execution_count
    with pytest.raises(ValueError, match="full-batch"):
        execution.execute_decode(invalid, (), now)
    _same([memory.summary() for memory in execution.memory], before)
    assert execution.layer_execution_count == layers


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("invalid", ["duplicate-slot", "input-length", "output-length"])
def test_cache_hit_cannot_bypass_stable_slots_or_homogeneous_lengths(enabled, invalid):
    execution = _engine(cached=enabled)
    valid = _requests((0, 1, 2, 3))
    now = execution.execute_decode(valid, (), _import_initial(execution, valid)).end_ns
    bad = list(valid)
    if invalid == "duplicate-slot":
        bad[1] = replace(bad[1], slot_id=0)
    elif invalid == "input-length":
        # Swap contexts between the first two requests: aggregate key is
        # unchanged, while one input length violates the configured workload.
        bad[0] = replace(bad[0], input_tokens=11)
        bad[1] = replace(bad[1], generated_decode_tokens=0)
    else:
        bad[0] = replace(bad[0], output_tokens=10)
    before = [memory.summary() for memory in execution.memory]
    with pytest.raises(ValueError, match="full-batch|lengths"):
        execution.execute_decode(tuple(bad), (), now)
    _same([memory.summary() for memory in execution.memory], before)


def test_ep_microbatch_and_chunk_sums_are_part_of_the_work_signature():
    parallel = PaperParallelism(ep=2, microbatches=2)
    execution = _engine(cached=True, parallel=parallel, batch=8,
                         model=_model(moe=True, chunk=8))
    a = _requests((0, 1, 2, 3, 0, 1, 2, 3))
    # Same global sum, different EP ownership within a microbatch.
    different_ep = _requests((1, 0, 2, 3, 0, 1, 2, 3))
    assert execution._decode_signature_and_capacity(a, ()) != execution._decode_signature_and_capacity(different_ep, ())
    # Same per-EP full-context sum, different modulo-chunk sum.
    before_chunk_wrap = _requests((0, 1, 6, 3, 0, 1, 2, 3))
    different_chunk = _requests((2, 1, 4, 3, 0, 1, 2, 3))
    first = execution._decode_signature_and_capacity(before_chunk_wrap, ())
    second = execution._decode_signature_and_capacity(different_chunk, ())
    assert first[::2] == second[::2]
    assert first[1::2] != second[1::2]


def test_related_future_pd_for_later_gpu_forces_a_whole_cache_miss_before_any_replay(monkeypatch):
    settings = dict(parallel=PaperParallelism(ep=2), batch=4, model=_model(moe=True, chunk=8))
    baseline, cached = _engine(cached=False, **settings), _engine(cached=True, **settings)
    a = _requests((0, 1, 2, 3))
    times = [_import_initial(actor, a) for actor in (baseline, cached)]
    for index, actor in enumerate((baseline, cached)):
        times[index] = actor.execute_decode(a, (), times[index]).end_ns
    assert cached._decode_cache
    imports = []
    for index, actor in enumerate((baseline, cached)):
        successor = ServingRequest(1001, 10, 9, slot_id=1)
        before_program = sum(memory.bytes_by_operation.get("hbf:write", 0) for memory in actor.memory)
        imported = actor.import_kv(successor, times[index] + 5_000_000)
        after_program = sum(memory.bytes_by_operation.get("hbf:write", 0) for memory in actor.memory)
        assert imported.physical_program_bytes == after_program - before_program > 0
        imports.append(imported)
    calls = []
    for rank, memory in enumerate(cached.memory):
        original = memory.apply_quiescent_transition
        def tracked(*args, _rank=rank, _original=original, **kwargs):
            calls.append(_rank)
            return _original(*args, **kwargs)
        monkeypatch.setattr(memory, "apply_quiescent_transition", tracked)
    hits, misses = cached.decode_cache_hits, cached.decode_cache_misses
    results = [actor.execute_decode(a, (pending,), now) for actor, pending, now in
               zip((baseline, cached), imports, times)]
    assert cached.decode_cache_hits == hits
    assert cached.decode_cache_misses == misses + 1
    assert not calls  # GPU 0 cannot replay before GPU 1 rejects the global gate.
    assert all(result.end_ns >= imported.ready_ns for result, imported in zip(results, imports))
    _same(asdict(results[0]), asdict(results[1]))
    _same_engine(cached, baseline)


def test_matching_reserved_pd_frontiers_hit_nonconsecutively_without_duplicating_pd_bytes():
    settings = dict(parallel=PaperParallelism(ep=2), batch=4, model=_model(moe=True, chunk=8))
    baseline, cached = _engine(cached=False, **settings), _engine(cached=True, **settings)
    a, b = _requests((0, 1, 2, 3)), _requests((1, 2, 3, 4), identity=100)
    times = [_import_initial(actor, a) for actor in (baseline, cached)]
    for index, (query, offset, expect_hit) in enumerate(((a, 5_000_000, False),
            (b, 8_000_000, False), (a, 5_000_000, True), (a, 5_004_096, False))):
        hits = cached.decode_cache_hits
        results = []
        for side, actor in enumerate((baseline, cached)):
            successor = ServingRequest(1001 + index, 10, 9, slot_id=1)
            before_pd = sum(memory.bytes_by_operation.get("hbf:write", 0) for memory in actor.memory)
            pending = actor.import_kv(successor, times[side] + offset)
            after_pd = sum(memory.bytes_by_operation.get("hbf:write", 0) for memory in actor.memory)
            assert after_pd - before_pd == pending.physical_program_bytes > 0
            result = actor.execute_decode(query, (pending,), times[side])
            after_decode = sum(memory.bytes_by_operation.get("hbf:write", 0) for memory in actor.memory)
            assert after_decode - after_pd == result.physical_program_bytes
            assert result.end_ns >= pending.ready_ns
            times[side] = result.end_ns
            results.append(asdict(result))
        assert cached.decode_cache_hits == hits + int(expect_hit)
        _same(results[0], results[1])
        _same_engine(cached, baseline)


def test_future_pd_network_receive_alone_blocks_all_gpu_replay_atomically(monkeypatch):
    # Stage 1 begins at rank 8, so PP traffic crosses the eight-GPU node
    # boundary and uses the same InfiniBand receive port as incoming PD.
    settings = dict(parallel=PaperParallelism(pp=2, tp=8), batch=2, model=_model(layers=4))
    baseline, cached = _engine(cached=False, **settings), _engine(cached=True, **settings)
    a = _requests((0, 1))
    times = [_import_initial(actor, a) for actor in (baseline, cached)]
    for index, actor in enumerate((baseline, cached)):
        times[index] = actor.execute_decode(a, (), times[index]).end_ns
    assert cached._decode_cache
    receive_ends = [actor.network.receive_pd(8, 4096, now + 5_000_000)
                    for actor, now in zip((baseline, cached), times)]
    key = cached._decode_signature_and_capacity(a, ())
    templates = cached._decode_cache[key][0]
    assert all(memory.can_apply_quiescent_transition(template, times[1], signature=key)
               for memory, template in zip(cached.memory, templates))
    calls = []
    for rank, memory in enumerate(cached.memory):
        original = memory.apply_quiescent_transition
        def tracked(*args, _rank=rank, _original=original, **kwargs):
            calls.append(_rank)
            return _original(*args, **kwargs)
        monkeypatch.setattr(memory, "apply_quiescent_transition", tracked)
    hits = cached.decode_cache_hits
    results = [actor.execute_decode(a, (), now) for actor, now in zip((baseline, cached), times)]
    assert cached.decode_cache_hits == hits
    assert not calls
    assert all(result.end_ns >= arrival for result, arrival in zip(results, receive_ends))
    _same(asdict(results[0]), asdict(results[1]))
    _same_engine(cached, baseline)


@pytest.mark.parametrize("hardware,parallel,moe,chunk,batch", [
    ("HBM4", PaperParallelism(tp=2), False, 0, 4),
    ("HBF+", PaperParallelism(ep=2), True, 8, 4),
    ("HBF+", PaperParallelism(pp=2, microbatches=2), False, 0, 4),
    ("HBF+", PaperParallelism(pp=2, ep=2, microbatches=2), True, 8, 8),
])
def test_closed_loop_fresh_admissions_pd_and_parallel_execution_match_with_cache_switch(
        hardware, parallel, moe, chunk, batch):
    settings = dict(hardware=hardware, parallel=parallel, model=_model(moe=moe, chunk=chunk),
                    batch=batch, output=5)
    baseline, cached = _engine(cached=False, **settings), _engine(cached=True, **settings)
    outputs = [actor.run(warmup_steps=9, measurement_steps=24) for actor in (baseline, cached)]
    for output in outputs:
        assert output["completed_requests"] > 0 and output["admitted_requests"] > 0
        assert output["pd_transfer_issued_bytes"] > 0 and output["startup_import_bytes"] > 0
        assert output["logical_decode_write_bytes"] > 0
        assert output["full_batch_step_fraction"] == 1
        assert output["old_single_request_fast_mode_used"] is False
        assert output["measured_steps"] == 24
        if hardware != "HBM4":
            assert output["pd_program_completed_bytes"] > 0 and output["decode_program_bytes"] > 0
        for stage in output["capacity"]["stages"]:
            assert stage["accepted"]
    # Cache diagnostics and its single configuration switch intentionally
    # differ; every measured serving/resource/capacity result must agree.
    for output in outputs:
        output.pop("decode_transition_cache_hits")
        output.pop("decode_transition_cache_misses")
        output["research_assumptions"].pop("use_decode_transition_cache")
    _same(outputs[0], outputs[1], "closed_loop")
    _same_engine(cached, baseline)
