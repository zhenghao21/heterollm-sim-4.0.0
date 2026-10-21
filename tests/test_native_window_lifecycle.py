"""Forecast batch identities reuse the native selector and packing rules."""
from dataclasses import replace

import pytest

from heterollm_sim.ir import RequestSpec
from heterollm_sim.native_window_lifecycle import (native_admission_washout,
    native_lifecycle_skeleton, native_layer_lifecycle_skeleton)
from test_native_external_kv import ingress_scenario, native_runtime


def lifecycle_scenario(batch, output=8):
    scenario = ingress_scenario("hbm_tp2", warmup=output-1, measured=2*(output-1), count=4*batch)
    requests = []
    for index in range(4*batch):
        age = index*(output-1)//batch if index < batch else 0
        requests.append(RequestSpec(f"r{index:04d}", 0., 7+age, output-age,
            metadata={"warmup_seed": index < batch}))
    return replace(scenario, workload=replace(scenario.workload, requests=tuple(requests),
        scheduler=replace(scenario.workload.scheduler, max_num_seqs=batch,
            max_num_batched_tokens=batch, max_num_ubatch_tokens=batch,
            prefill_chunk_tokens=1, policy="decode_first", preemption_enabled=False,
            mixed_phase_batching=False)))


def capture_first_native_query(scenario):
    from heterollm_sim.execution_control import _NativeLayerSamplingCutoff

    runtime, prepared = native_runtime(scenario), {}
    def capture(active, cohort, cost, stages, ready):
        assert active is runtime
        prepared.update(cohort=cohort, cost=cost, stages=stages, ready=ready)
        raise _NativeLayerSamplingCutoff()
    runtime._native_layer_sampler = capture
    with pytest.raises(_NativeLayerSamplingCutoff):
        runtime.run_until_decode_steps(1, strict_cohort_cutoff=True)
    return runtime, prepared


def test_real_b8_prequery_layer_lifecycle_never_invents_a_completed_cohort():
    runtime, prepared = capture_first_native_query(lifecycle_scenario(8))
    times = (runtime.now, runtime._completed_decode_steps, len(runtime.events))
    states = tuple((row.status, row.committed, row.prefill_cursor, row.slot_id)
                   for row in runtime._state_values)
    timeline = runtime._execution_resource_kernel.physical_runtime.snapshot()["timeline"]
    skeleton = native_layer_lifecycle_skeleton(runtime, prepared["cohort"])
    assert not runtime.batches
    assert times == (runtime.now, runtime._completed_decode_steps, len(runtime.events))
    assert states == tuple((row.status, row.committed, row.prefill_cursor, row.slot_id)
                           for row in runtime._state_values)
    assert timeline == runtime._execution_resource_kernel.physical_runtime.snapshot()["timeline"]
    assert skeleton["physical_complete_cohort_samples"] == skeleton["physical_visible_output_tokens"] == 0
    assert skeleton["events"][0]["forecast_only"]
    assert not skeleton["events"][0]["observed_prefix"]
    assert skeleton["seed_washout"]["qualified"]
    assert not skeleton["seed_washout"]["per_request_label_order_proven"]
    assert len(skeleton["complete_fresh"]) == 16
    del runtime._native_layer_sampler
    actual = runtime.run()
    assert [(row["kind"], row["cohort_id"], row["request_ids"], row["items"])
            for row in skeleton["events"]] == [(cohort.kind, cohort.cohort_id,
        tuple(cohort.request_ids), tuple({"request_id": item.request_id, "phase": item.phase,
            "token_count": item.token_count, "context_tokens": item.context_tokens,
            "committed_tokens": item.committed_tokens} for item in cohort.items))
        for cohort in actual.batches]
    assert skeleton["visible_output_tokens"] == actual.decode_measurement["visible_output_tokens"]
    waves = {(row["first_event"], row["last_event"]) for row in skeleton["complete_fresh"]}
    for first, last in waves:
        members = skeleton["events"][first]["request_ids"]
        start_ns = min(actual.request_metrics[key].first_token_ns for key in members)
        end_ns = max(actual.request_metrics[key].last_token_ns for key in members)
        starts = [event for event in actual.events if event.event_type == "external_kv_import_start"
                  and start_ns < event.timestamp_ns < end_ns]
        assert len(starts) <= runtime._external_ingress["max_inflight_requests"]


@pytest.mark.parametrize("batch", (1, 2, 8))
def test_logical_skeleton_matches_real_native_tp2_pd_batch_items_and_complete_tokens(batch):
    runtime = native_runtime(lifecycle_scenario(batch))
    runtime.run_until_decode_steps(5, strict_cohort_cutoff=True)
    before = runtime._execution_resource_kernel.physical_runtime.snapshot()
    skeleton = native_lifecycle_skeleton(runtime, 7, 14)
    after = runtime._execution_resource_kernel.physical_runtime.snapshot()
    assert before["timeline"] == after["timeline"]
    assert not skeleton["physical_state_advanced"] and not skeleton["timestamps_generated"]
    full = runtime.run()
    observed = [{"kind": cohort.kind, "cohort_id": cohort.cohort_id,
        "request_ids": tuple(cohort.request_ids),
        "items": tuple({"request_id": item.request_id, "phase": item.phase,
            "token_count": item.token_count, "context_tokens": item.context_tokens,
            "committed_tokens": item.committed_tokens} for item in cohort.items)} for cohort in full.batches]
    assert [{key: row[key] for key in ("kind", "cohort_id", "request_ids", "items")}
            for row in skeleton["events"]] == observed
    assert skeleton["visible_output_tokens"] == full.decode_measurement["visible_output_tokens"]
    assert {row["request_id"] for row in skeleton["complete_fresh"]} == set(
        full.decode_measurement["complete_request_tpot_ns"])
    if batch == 8:
        # Filled runnable slots include imported queries. They do not imply
        # that the native Decode selector has eight eligible sequences.
        assert {len(row["request_ids"]) for row in skeleton["events"]
                if row["kind"] == "decode"} >= {2, 4, 8}
        assert skeleton["fresh_wave_shapes_qualified_after_seed_washout"]
        assert not skeleton["seed_washout"]["per_request_label_order_proven"]
    if batch == 2:
        assert not skeleton["fresh_wave_shapes_qualified_after_seed_washout"]
        assert "warmup_seed_still_waiting_or_prefilling" in skeleton["seed_washout"]["reasons"]


@pytest.mark.parametrize("policy", ("decode_first_aging", "fcfs"))
def test_time_dependent_or_unknown_scheduler_is_not_silently_approximated(policy):
    runtime = native_runtime(lifecycle_scenario(2))
    runtime.plan = replace(runtime.plan, scheduler=replace(runtime.plan.scheduler, policy=policy))
    with pytest.raises(ValueError, match="closed_time_independent"):
        native_lifecycle_skeleton(runtime, 7, 14)


def test_unimported_aged_seed_is_not_assumed_fifo_before_shorter_fresh_request():
    runtime = native_runtime(lifecycle_scenario(2))
    # The first real native batch can contain r0+r2: the longer r1 seed's
    # physical PD transfer may still be in flight. No all-ready shortcut.
    with pytest.raises(ValueError, match="heterogeneous_unimported"):
        native_lifecycle_skeleton(runtime, 7, 14)


def test_fresh_wave_washout_does_not_assume_unqualified_seed_or_token_budget():
    runtime = native_runtime(lifecycle_scenario(8))
    assert not native_admission_washout(runtime)["qualified"]
    runtime.run_until_decode_steps(5, strict_cohort_cutoff=True)
    assert native_admission_washout(runtime)["qualified"]
    runtime.plan = replace(runtime.plan, scheduler=replace(runtime.plan.scheduler, max_num_batched_tokens=1))
    rejected = native_admission_washout(runtime)
    assert not rejected["qualified"]
    assert "fresh_query_cannot_form_full_single_token_wave" in rejected["reasons"]


def test_equal_size_native_pd_imports_need_not_finish_in_submission_order():
    from heterollm_sim.ir import ComponentSpec
    from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
    from heterollm_sim.memory_types import AccessRequest, NandConfig, SramConfig

    config = NandConfig(kind="HBF", channels=2, planes_independent=True,
        page_bytes=4096, host_bandwidth_gb_s=100., host_read_bandwidth_gb_s=100.,
        host_write_bandwidth_gb_s=100., internal_bandwidth_gb_s=100.,
        page_read_ns=10., page_program_ns=100., front_ns=0., capacity_bytes=65536,
        host_full_duplex=True)
    executor = PhysicalKVTransferExecutor([
        ComponentSpec("src", "sram", capacity_bytes=65536, metadata={
            "physical_memory_config": SramConfig(65536, 64, 100., 100., 100., 1., 1., 2, 8)}),
        ComponentSpec("dst", "hbf", capacity_bytes=65536,
            metadata={"physical_memory_config": config})], retain_phase_details=False)
    owner = executor._ensure_service("dst").physical_owner
    allocator = executor._allocator(owner)
    allocator.allocate("a", 4096)
    allocator.allocate("b", 4096)
    # Real prior programming leaves a's array busy; b maps to another
    # independent array. This is ordinary native Core state, not an override.
    executor.runtime.runtimes[owner].core.submit(AccessRequest("busy", "write", 0, 4096, 0.))
    plans, completed = {}, {}
    for request_id in ("a", "b"):
        plans[request_id] = executor.prepare_external_import(source_component="src",
            destinations=[{"component_id": "dst", "buffer_id": request_id,
                "generation": 0, "offset_bytes": 0, "size_bytes": 4096}],
            buffer_bytes=4096, chunk_bytes=4096, request_id=request_id, arrival_ns=1.)
    while plans:
        key = min(plans, key=lambda value: (plans[value].completion_ns, value))
        phase = plans[key]
        if phase.advance is None:
            completed[key] = phase.completion_ns
            phase.commit()
            del plans[key]
        else:
            plans[key] = phase.advance()
    assert completed["b"] < completed["a"]
    assert executor.counters["read_bytes"] == executor.counters["write_bytes"] == 8192
