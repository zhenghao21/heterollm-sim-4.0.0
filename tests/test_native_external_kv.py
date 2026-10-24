"""External KV uses native physical buffers, readiness, and serving events."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from exploring_hbf_native_pilot import build_pilot_scenario
from heterollm_sim.control_plane import bootstrap_control_plane
from heterollm_sim.execution_control import ExecutionCancelledError, ExecutionControl
from heterollm_sim.ir import RequestSpec
from heterollm_sim.planner import (TopologyAwareBatchCostProvider,
    compile_serving_cohort_schedule, native_kv_ingress_inventory)
from heterollm_sim.reporting import report_dict, run_scenario
from heterollm_sim.serving import (BatchCohort, BatchItem, RequestStatus, _OnlineRuntime,
    compile_serving_plan)


def ingress_scenario(case="hbm_tp1", *, warmup=0, measured=0, count=4):
    scenario = build_pilot_scenario(case)
    rank_map = {str(rank.rank): "hbf0" if case == "hbf_weights_kv_tp1" else rank.memory_component_id
                for rank in scenario.placement.parallel.rank_mapping}
    return replace(scenario,
        placement=replace(scenario.placement, metadata={**scenario.placement.metadata,
            "rank_kv_components": rank_map}),
        workload=replace(scenario.workload, requests=tuple(RequestSpec(f"r{i}",0.,7,4) for i in range(count)),
            request_count=count, metadata={**scenario.workload.metadata,
                "external_kv_ingress": {"source_component":"hostmem0", "buffer_bytes":4096,
                    "chunk_bytes":1024,"max_inflight_requests":2,"warmup_decode_cohorts":warmup,
                    "measurement_decode_cohorts":measured,"finish_after_measurement":True}}))


def native_runtime(scenario):
    bootstrap = bootstrap_control_plane(scenario,capture_physical_details=False)
    scenario = bootstrap.scenario
    return _OnlineRuntime(compile_serving_plan(scenario),
        TopologyAwareBatchCostProvider(scenario,defer_physical_execution=True),
        execution_kernel=bootstrap.kernel,runtime_origin_ns=bootstrap.makespan_ns)


def test_import_is_not_admitted_or_given_free_output_before_target_completion():
    runtime = native_runtime(ingress_scenario(count=2))
    runtime._arrive()
    runtime._start_external_imports()
    assert len(runtime._external_pending) == 2
    assert runtime._external_executor.counters["read_bytes"] > 0
    assert runtime._external_executor.counters["write_bytes"] == 0
    runtime._fill_slots()
    assert runtime._running_count == 0
    assert all(state.prefill_cursor == state.committed == 0 for state in runtime.states.values())
    while runtime._external_pending:
        runtime.now = runtime._next_external_phase()
        runtime._complete_external_imports()
    assert all(state.prefill_cursor == 6 and state.committed == 0 for state in runtime.states.values())
    runtime._fill_slots()
    cohort = runtime._select_cohort()
    assert cohort.kind == "prefill"
    assert all(item.context_tokens == 6 and item.token_count == 1 for item in cohort.items)
    result = runtime.run()
    assert all(metric.status is RequestStatus.FINISHED and metric.visible_output_tokens == 4
               for metric in result.request_metrics.values())
    assert result.external_kv_ingress_metrics["imported_logical_bytes"] == 2*6*2*2*256
    assert result.external_kv_ingress_metrics["source_buffer_peak_bytes"] <= 4096
    assert result.external_kv_ingress_metrics["counters"]["read_bytes"] == result.external_kv_ingress_metrics["counters"]["write_bytes"]


def test_cancel_drains_submitted_phase_and_retains_real_source_read_cost():
    runtime = native_runtime(ingress_scenario(count=2))
    runtime._arrive()
    runtime._start_external_imports()
    read = runtime._external_executor.counters["read_bytes"]
    assert read > 0
    runtime._cancel_external_imports()
    assert not runtime._external_pending
    assert runtime._external_executor.counters["read_bytes"] == read
    assert runtime._external_executor.counters["write_bytes"] == 0
    assert runtime._external_executor.counters["cancelled_transfer_count"] == 2
    assert runtime._execution_resource_kernel.makespan_ns >= runtime.now
    assert runtime.ledger.used_pages == 0
    assert all(state.prefill_cursor == state.committed == 0 for state in runtime.states.values())
    assert not runtime._execution_resource_kernel._request_physical_buffers


def test_failed_preparation_rolls_back_capacity_and_native_buffer_ownership(monkeypatch):
    runtime = native_runtime(ingress_scenario(count=2))
    runtime._arrive()
    monkeypatch.setattr(runtime._external_executor,"prepare_external_import",
                        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("injected source rejection")))
    with pytest.raises(RuntimeError,match="injected source rejection"):
        runtime._start_external_imports()
    assert runtime.ledger.used_pages == 0
    assert not runtime._external_pending
    assert not runtime._execution_resource_kernel._request_physical_buffers
    assert runtime._external_executor.counters["read_bytes"] == 0


def test_fixed_decode_window_reuses_runtime_and_does_not_fake_queue_completion():
    result = run_scenario(ingress_scenario(warmup=3,measured=3,count=8))
    window = result.serving.decode_measurement
    assert window["complete"] and window["measured_decode_cohorts"] == 3
    assert window["batch_sizes"] == [2,2,2]
    assert window["complete_fresh_requests"] == 2
    assert window["complete_request_tpot_ns"] and all(value > 0 for value in window["complete_request_tpot_ns"].values())
    assert any(metric.status is RequestStatus.WAITING for metric in result.serving.request_metrics.values())
    assert all(item.token_count == 1 for batch in result.serving.batches if batch.kind == "prefill" for item in batch.items)
    assert report_dict(result,include_visualization=False)["summary"]["decode_measurement"] == window
    assert window["duration_ns"] > sum(batch.end_ns-batch.start_ns for batch in result.serving.batches
        if batch.kind == "decode" and window["start_ns"] < batch.end_ns <= window["end_ns"])


def test_native_tp2_imports_both_private_rank_kv_buffers():
    result = run_scenario(ingress_scenario("hbm_tp2",count=2))
    ledger = result.serving.external_kv_ingress_metrics["cost_ledger"]
    assert ledger["owners"]["hbm0.hbm_fabric"]["physical_write_bytes"] > 0
    assert ledger["owners"]["hbm1.hbm_fabric"]["physical_write_bytes"] > 0
    assert set(result.serving.plan.kv_policy.component_bytes_per_page) == {"hbm0","hbm1"}
    assert sum(result.serving.plan.kv_policy.component_bytes_per_page.values()) == result.serving.plan.kv_policy.bytes_per_page
    assert all(metric.visible_output_tokens == 4 for metric in result.serving.request_metrics.values())


def test_native_hbf_import_pays_nand_program_before_first_real_output():
    result = run_scenario(ingress_scenario("hbf_weights_kv_tp1",count=2))
    ingress = result.serving.external_kv_ingress_metrics
    assert ingress["cost_ledger"]["storage_traffic"]["physical_write_bytes"] > 0
    assert ingress["cost_ledger"]["dram_traffic"]["physical_read_bytes"] > 0
    done = {event.request_id:event.timestamp_ns for event in result.serving.events
            if event.event_type == "external_kv_import_complete"}
    assert all(metric.first_token_ns > done[key] for key,metric in result.serving.request_metrics.items())


def test_chunked_pd_import_uses_recent_nonzero_logical_block_and_crosses_boundaries():
    from test_native_maverick_semantics import miniature_maverick
    scenario = miniature_maverick(prompt=7)
    authored = ingress_scenario(count=1)
    scenario = replace(scenario, placement=replace(scenario.placement,
        metadata={**scenario.placement.metadata,"rank_kv_components":{"0":"hbm0"}}),
        workload=replace(scenario.workload,metadata=authored.workload.metadata))
    result = run_scenario(scenario)
    # Three C=4 layers import positions [4,6); one full layer imports [0,6).
    assert result.serving.external_kv_ingress_metrics["imported_logical_bytes"] == (3*2+6)*128
    metric, = result.serving.request_metrics.values()
    assert metric.status is RequestStatus.FINISHED and metric.visible_output_tokens == 5
    assert result.serving.plan.kv_policy.kv_layer_attention_limits == {"dense0":4,"dense1":4,"dense2":4}


def test_no_external_mode_keeps_original_runtime_and_empty_measurement():
    result = run_scenario(build_pilot_scenario("hbm_tp1"))
    assert not result.serving.external_kv_ingress_metrics
    assert not result.serving.decode_measurement
    assert any(item.token_count == 32 for batch in result.serving.batches for item in batch.items)


@pytest.mark.parametrize("case", ["hbm_tp1", "hbm_tp2", "hbf_weights_kv_tp1", "chunked"])
def test_inventory_matches_native_full_graph_declarations_without_compiling_operator_dag(case, monkeypatch):
    import heterollm_sim.planner as planner
    if case == "chunked":
        from test_native_maverick_semantics import miniature_maverick
        scenario = miniature_maverick(prompt=7)
    else:
        scenario = ingress_scenario(case, count=1)
    request = scenario.workload.requests[0]
    history = request.prompt_tokens-1
    item = BatchItem(request.request_id,"prefill",1,history,kv_append_tokens=1,
        kv_materialized_tokens=1,main_tokens=0,draft_tokens=0,verifier_tokens=1,
        committed_tokens=0,logit_tokens=1,completion_cursor=request.prompt_tokens)
    schedule = compile_serving_cohort_schedule(scenario,BatchCohort("inventory-proof","prefill",0.,(item,)))
    expected, contracts = {}, []
    for task in schedule.tasks:
        for row in task.metadata.get("physical_allocations", ()):
            if str(row["buffer_id"]).startswith("kv:" + request.request_id + ":"):
                owner = row.get("physical_owner") or task.metadata["physical_owner"]
                expected[(owner,row["buffer_id"],row.get("generation",0))] = {**row,"physical_owner":owner}
        contract = task.metadata.get("generic_kv_storage")
        if contract is not None and contract not in contracts:
            contracts.append(contract)
    monkeypatch.setattr(planner,"_lower_serving_cohort",lambda *_a,**_k: pytest.fail("inventory compiled operator DAG"))
    inventory = native_kv_ingress_inventory(scenario,request.request_id,history,ep_group=0)
    actual = {(row["physical_owner"],row["buffer_id"],row.get("generation",0)):row
              for row in inventory["physical_allocations"]}
    assert actual == expected
    assert len(inventory["contracts"]) == len(contracts)
    assert all(contract in contracts for contract in inventory["contracts"])


def test_inventory_cancellation_is_reachable_inside_rank_loop(monkeypatch):
    import heterollm_sim.planner as planner
    scenario = ingress_scenario("hbm_tp2",count=1)
    contract = planner._generic_kv_contract
    calls = []
    def record(*args,**kwargs):
        calls.append(args[3].rank)
        return contract(*args,**kwargs)
    monkeypatch.setattr(planner,"_generic_kv_contract",record)
    control = ExecutionControl(cancellation_callback=lambda: bool(calls))
    with pytest.raises(ExecutionCancelledError):
        native_kv_ingress_inventory(scenario,"r0",6,ep_group=0,execution_control=control)
    assert calls == [0]


def test_external_compact_phase_ledger_keeps_real_totals_and_commit_is_idempotent():
    from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
    def submitted(retain):
        runtime = native_runtime(ingress_scenario("hbm_tp2",count=1))
        executor = PhysicalKVTransferExecutor(runtime.plan.scenario.hardware.components,
            runtime=runtime._execution_resource_kernel.physical_runtime,
            topology=runtime._external_executor.topology,namespace="comparison",
            retain_phase_details=retain)
        runtime._external_executor = executor
        destinations = runtime._external_destination_ranges(runtime.states["r0"])
        plan = executor.prepare_external_import(source_component="hostmem0",destinations=destinations,
            buffer_bytes=512,chunk_bytes=256,request_id="r0",arrival_ns=runtime.now)
        while plan.advance is not None:
            plan = plan.advance()
        event = plan.commit()
        assert plan.commit() == event
        with pytest.raises(Exception,match="different outcome"):
            plan.cancel()
        return executor, plan.completion_ns
    full, full_end = submitted(True)
    compact, compact_end = submitted(False)
    assert full_end == compact_end
    assert full.counters == compact.counters
    def physical_state(context):
        state = context.snapshot()
        # Contexts own distinct Core/allocator instances. Compare their state,
        # not those instance identities retained for same-context restoration.
        return {"timeline": state["timeline"], "allocations": state["allocator_state"],
            "committed_owners": state["committed_owners"],
            "cores": {owner:{key:value for key,value in row.items() if key != "active"}
                for owner,row in state["runtime_state"].items()}}
    assert physical_state(full.runtime) == physical_state(compact.runtime)
    detailed, bounded = full.cost_ledger(), compact.cost_ledger()
    assert bounded["phase_count"] == detailed["phase_count"] > bounded["phase_group_count"]
    assert bounded["phase_group_count"] <= 8
    assert not bounded["phases"] and not bounded["phase_detail_retained"]
    assert bounded["owners"] == detailed["owners"]
    assert bounded["links"] == detailed["links"]
    assert bounded["resource_bytes"] == detailed["resource_bytes"]
    for key in ("dram_traffic", "storage_traffic"):
        for field in ("task_count","logical_bytes","physical_bytes","physical_read_bytes","physical_write_bytes"):
            assert bounded[key].get(field) == detailed[key].get(field)
    snapshot = compact.snapshot()
    compact._cost_groups.clear()
    compact.restore(snapshot)
    assert compact.cost_ledger() == bounded


@pytest.mark.parametrize("prompt", (1, 5))
def test_all_chunk_layers_at_fresh_chunk_boundary_import_nothing_but_execute_first_query(prompt):
    from heterollm_sim.ir import ModelSpec, build_model_graph_from_layer_specs, model_graph_execution_layers
    from test_native_maverick_semantics import miniature_maverick
    scenario = miniature_maverick(prompt=prompt)
    layers = [replace(layer,sequence_mixer="chunked_attention",attention_chunk_size=4,
        use_rope=True) for layer in model_graph_execution_layers(scenario.model.graph)]
    model = ModelSpec(name=scenario.model.name,graph=build_model_graph_from_layer_specs(
        scenario.model.name,layers,architecture="llama4",vocabulary_size=128,
        max_sequence_length=16384,embedding_weight_bytes=64*128*2,
        output_weight_bytes=64*128*2,output_head_dtype="bf16"))
    scenario = replace(scenario,model=model,workload=replace(scenario.workload,
        metadata=ingress_scenario(count=1).workload.metadata))
    inventory = native_kv_ingress_inventory(scenario,"native-mav",prompt-1,ep_group=0)
    assert len(inventory["physical_allocations"]) == 8
    result = run_scenario(scenario)
    assert result.serving.external_kv_ingress_metrics["imported_logical_bytes"] == 0
    assert result.serving.external_kv_ingress_metrics["counters"]["read_bytes"] == 0
    metric = result.serving.request_metrics["native-mav"]
    assert metric.status is RequestStatus.FINISHED and metric.visible_output_tokens == 5
    first = result.serving.batches[0]
    assert first.kind == "prefill" and first.items[0].context_tokens == prompt-1
    assert first.items[0].token_count == 1 and first.end_ns > first.start_ns


def test_unknown_external_price_failure_protects_other_owners_and_preserves_source_cost(monkeypatch):
    from heterollm_sim.data_motion import PhysicalService
    runtime = native_runtime(ingress_scenario(count=1))
    runtime._arrive()
    runtime._start_external_imports()
    executor = runtime._external_executor
    plan = executor._async_plan_states[next(iter(executor._async_plan_states))]
    transfer = runtime._external_pending["r0"]["plan"]
    while plan["actions"][plan["action"]][0] != "memory":
        transfer = transfer.advance()
    before = executor.runtime.snapshot()
    counters = dict(executor.counters)
    ledger = executor.cost_ledger()
    assert counters["read_bytes"] > 0 and counters["write_bytes"] == 0
    saved_snapshot, saved_price = executor.runtime.snapshot, PhysicalService._price
    owners = []
    def snapshot(*,transactional=False,owners=None,_owned_timeline=False):
        # An unknown price callback can mutate another owner; it must retain
        # the complete public undo rather than a private one-owner journal.
        assert not transactional and owners is None
        assert _owned_timeline is False
        return saved_snapshot(transactional=transactional,owners=owners,_owned_timeline=_owned_timeline)
    def rejected(self,*args,**kwargs):
        result = saved_price(self,*args,**kwargs)
        owners.append(self.physical_owner)
        source_owner = executor.services["hostmem0"].physical_owner
        executor.runtime.runtimes[source_owner].clock_ns += 1234.
        raise RuntimeError("injected target failure after real submit")
    monkeypatch.setattr(executor.runtime,"snapshot",snapshot)
    monkeypatch.setattr(PhysicalService,"_price",rejected)
    with pytest.raises(RuntimeError,match="injected target failure"):
        transfer.advance()
    assert owners == ["hbm0.hbm_fabric"]
    assert executor.counters == counters and executor.cost_ledger() == ledger
    assert saved_snapshot() == before
    assert plan["phase_token"] == executor._transfer_state(plan)["phase_token"]


def test_native_external_phase_requests_owned_timeline_but_public_access_does_not(monkeypatch):
    runtime = native_runtime(ingress_scenario(count=1))
    executor = runtime._external_executor
    original = executor.runtime.begin_transaction
    flags = []
    def begin(*, owners=None, _owned_timeline=False):
        flags.append((owners, _owned_timeline))
        return original(owners=owners, _owned_timeline=_owned_timeline)
    monkeypatch.setattr(executor.runtime, "begin_transaction", begin)
    runtime._arrive()
    runtime._start_external_imports()
    assert flags and all(owners is not None and owned for owners, owned in flags)
    with executor._io_transaction(owners=flags[-1][0]):
        pass
    assert flags[-1][1] is False


@pytest.mark.parametrize("case", ("hbm_tp1", "hbf_weights_kv_tp1"))
def test_each_external_phase_exposes_every_native_changed_lane_including_link_credits(case):
    runtime = native_runtime(ingress_scenario(case, count=1))
    runtime._arrive()
    runtime._start_external_imports()
    plan = runtime._external_pending["r0"]["plan"]
    executor = runtime._external_executor
    assert plan.native_changed_resource_ids
    source_read_ids = set(plan.native_changed_resource_ids)
    observed_link_credit = False
    while plan.advance is not None:
        before = {name: tuple(lanes) for name, lanes in executor.runtime.timeline.lane_available.items()}
        plan = plan.advance()
        current = executor.runtime.timeline.lane_available
        changed = {name for name in before.keys() | current.keys()
                   if before.get(name) != tuple(current.get(name, ()))}
        if plan.native_changed_resource_ids is None:
            # A cold JEDEC controller has no resident column journal yet.
            # It must use complete publication until its native layout exists.
            assert case == "hbm_tp1"
        else:
            assert changed <= set(plan.native_changed_resource_ids)
            observed_link_credit |= any(name.endswith(".inflight") for name in plan.native_changed_resource_ids)
    assert observed_link_credit
    assert plan.native_changed_resource_ids is None or source_read_ids != set(plan.native_changed_resource_ids)


def test_authored_external_link_callback_keeps_complete_calendar_publication(monkeypatch):
    runtime = native_runtime(ingress_scenario(count=1))
    runtime._arrive()
    runtime._start_external_imports()
    executor = runtime._external_executor
    plan = runtime._external_pending["r0"]["plan"]
    original = executor._submit_link_phase
    def authored(*args, **kwargs):
        bill = original(*args, **kwargs)
        executor.runtime.timeline.lane_available["authored:foreign"] = [1234.]
        return bill
    monkeypatch.setattr(executor, "_submit_link_phase", authored)
    while plan.advance is not None:
        plan = plan.advance()
        if "authored:foreign" in executor.runtime.timeline.lane_available:
            assert plan.native_changed_resource_ids is None
            break
    else:
        pytest.fail("expected an actual link phase")


@pytest.mark.parametrize("target, name", (("executor", "_advance_external_phase"), ("runtime", "snapshot")))
def test_publication_support_rejects_authored_plan_producer_method(monkeypatch, target, name):
    from heterollm_sim.kv_transfer import native_external_publication_supported
    runtime = native_runtime(ingress_scenario(count=1))
    executor = runtime._external_executor
    assert native_external_publication_supported(executor)
    instance = executor if target == "executor" else executor.runtime
    original = getattr(instance, name)
    monkeypatch.setattr(instance, name, lambda *args, **kwargs: original(*args, **kwargs))
    assert not native_external_publication_supported(executor)


def test_owner_transaction_removes_new_runtime_and_allocator_on_failure():
    runtime = native_runtime(ingress_scenario(count=1))._external_executor.runtime
    config = runtime.runtimes["hbm0.hbm_fabric"].core.config
    before = runtime.snapshot()
    point = runtime.begin_transaction(owners=("new-owner",))
    runtime.runtime(config,"new-owner")
    runtime.allocators["new-owner"].allocate("new-buffer",128,0)
    runtime.restore(point)
    assert "new-owner" not in runtime.runtimes and "new-owner" not in runtime.allocators
    assert runtime.snapshot() == before


def test_strict_decode_cutoff_stops_with_queued_requests_and_resumes_exactly():
    scenario = ingress_scenario(count=4)
    expected = native_runtime(scenario).run()
    runtime = native_runtime(scenario)
    assert runtime.run_until_decode_steps(1, strict_cohort_cutoff=True) is False
    assert runtime._completed_decode_steps == 1
    assert any(state.status is not RequestStatus.RUNNING
               and state.status is not RequestStatus.FINISHED for state in runtime.states.values())
    prefix = runtime._result()
    assert not prefix.decode_measurement.get("complete", False)
    assert sum(row.status is RequestStatus.FINISHED for row in prefix.request_metrics.values()) == 0
    observed = runtime.run()
    assert observed.events == expected.events
    assert observed.request_metrics == expected.request_metrics
    assert observed.external_kv_ingress_metrics == expected.external_kv_ingress_metrics
    assert [(row.start_ns, row.end_ns) for row in observed.batches] == [
        (row.start_ns, row.end_ns) for row in expected.batches]
