"""Native forecast bounds come from authored resources, never certificates."""
from types import SimpleNamespace as NS

import pytest

from heterollm_sim.contracts import (_PreparedExecutionStage, _PreparedExecutionTask,
                                    ResourceDemand)
from heterollm_sim.native_window_producer import (_rank_private_phase_bound,
    _cold_resident_nand_weights, summarize_native_stages)


def task(identity, rank=None, dependencies=(), resources=(), metadata=None):
    values = {} if rank is None else {"rank": rank}
    values.update(metadata or {})
    return _PreparedExecutionTask(identity, tuple(dependencies), ("r0",),
        tuple(ResourceDemand(resource, 1.) for resource in resources), metadata=values)


def test_private_rank_branches_keep_their_entire_sum_and_parallelize_only_distinct_resources():
    rows = [(task("a", 0), 3., {"gpu0.l2"}),
            (task("b", 0), 5., {"gpu0.l2"}),
            (task("c", 1), 7., {"gpu1.l2"}),
            (task("d", 1), 11., {"gpu1.l2"})]
    value, proof = _rank_private_phase_bound(rows)
    assert value == 18.  # max(3+5,7+11), not max(each Task) or sum(26).
    assert proof[0]["qualified"] and proof[0]["rank_count"] == 2


@pytest.mark.parametrize("shared,cross", [(True, False), (False, True)])
def test_shared_resources_or_cross_rank_edges_keep_the_serial_phase(shared, cross):
    rows = [(task("a", 0), 3., {"shared" if shared else "gpu0.l2"}),
            (task("b", 1, ("a",) if cross else ()), 7., {"shared" if shared else "gpu1.l2"})]
    value, proof = _rank_private_phase_bound(rows)
    assert value == 10.
    assert proof[0]["qualified"] is (not shared)


def test_cross_rank_dependency_is_kept_without_serializing_unrelated_rank_work():
    rows = [(task("a", 0), 3., {"gpu0"}),
            (task("b", 1, ("a",)), 7., {"gpu1"}),
            (task("c", 2), 13., {"gpu2"})]
    value, proof = _rank_private_phase_bound(rows)
    assert value == 13.  # max(a+b,c), while full serial would be 23.
    assert proof[0]["cross_rank_dependencies"] and proof[0]["qualified"]


def test_host_and_collective_jobs_are_charged_and_separate_private_rank_blocks():
    rows = [(task("a", 0), 3., {"gpu0"}), (task("b", 1), 7., {"gpu1"}),
            (task("ring"), 13., {"link.shared"}),
            (task("c", 0), 17., {"gpu0"}), (task("d", 1), 19., {"gpu1"})]
    value, proof = _rank_private_phase_bound(rows)
    assert value == 7.+13.+19.
    assert proof[-1]["basis"] == "unranked_communication_and_shared_host_serial_sum"


def test_zero_time_private_marker_does_not_force_serialization_of_other_ranks():
    rows = [(task("a", 0), 3., {"gpu0"}),
            (task("a.done", dependencies=("a",)), 0., set()),
            (task("b", 1), 7., {"gpu1"})]
    assert _rank_private_phase_bound(rows)[0] == 7.


def test_zero_time_join_is_a_real_cross_rank_boundary():
    rows = [(task("a", 0), 3., {"gpu0"}), (task("b", 1), 7., {"gpu1"}),
            (task("join", dependencies=("a", "b")), 0., set()),
            (task("c", 0), 11., {"gpu0"}), (task("d", 1), 13., {"gpu1"})]
    assert _rank_private_phase_bound(rows)[0] == 7.+13.


def test_unknown_prepared_stage_cannot_supply_a_trusted_ir_summary():
    summary = summarize_native_stages((NS(execution_tasks=()),), NS())
    assert summary["unsupported"] == ["non_native_prepared_stage"]


def test_workload_metadata_certificate_is_not_read_as_physical_proof():
    stage = _PreparedExecutionStage("body", 0, (), ("r0",), "gpu0", 0.,
        (task("a", 0, resources=("gpu0.tensor",)),))
    hostile = NS(workload=NS(metadata={"native_window_budget": {
        "qualified": True, "row_phase_ns": 0., "fixed_path_floor_ns": 1e30}}))
    result = summarize_native_stages((stage,), hostile)
    assert result["fixed_path_floor_ns"] == 1.
    assert result["row_phase_per_cohort_ns"] == 0.


@pytest.mark.parametrize("pages_per_bank", (3, 5))
def test_native_full_resident_nand_scan_cannot_hit_its_previous_cached_suffix(pages_per_bank):
    from heterollm_sim.memory_allocator import PhysicalAddressAllocator
    from heterollm_sim.memory_types import AccessRequest, NandConfig, Operation
    from heterollm_sim.nand_core import NandCore

    config = NandConfig(kind="HBF", channels=2, page_bytes=4096,
        read_cache_pages_per_bank=2, capacity_bytes=1024**2,
        page_read_ns=10., page_program_ns=100.)
    allocator = PhysicalAddressAllocator(config.effective_capacity_bytes, config.page_bytes)
    allocation = allocator.allocate("weight", 2*pages_per_bank*4096, allocation_arena="resident")
    core = NandCore(config, capture_details=False)
    access = {"operation": "read", "physical_owner": "hbf", "buffer_id": "weight",
              "allocation_generation": 0, "offset_bytes": 0, "byte_count": allocation.size_bytes}
    md = {"physical_owner": "hbf", "memory_accesses": (access,),
          "weight_tensor_id": "model.weight", "weight_lifecycle_mode": "preloaded_resident"}
    row = task("scan", 0, metadata=md)
    stage = _PreparedExecutionStage("body", 0, (), ("r0",), "gpu0", 0., (row,))
    runtime = NS(allocators={"hbf": allocator}, runtimes={"hbf": NS(core=core)})
    assert _cold_resident_nand_weights((stage,), runtime) == {("hbf", "weight", 0)}
    first = core.submit(AccessRequest("first", Operation.READ, allocation.base_address, allocation.size_bytes))
    second = core.submit(AccessRequest("second", Operation.READ, allocation.base_address,
                                    allocation.size_bytes, arrival_ns=first.completion_ns))
    assert first.counters["read_cache_hits"] == second.counters["read_cache_hits"] == 0


def test_unused_nand_core_is_not_an_hbf_candidate():
    from heterollm_sim.native_window_budget import _memory_kind
    scenario = NS(placement=NS(metadata={"rank_kv_components": {"0": "hbm0"}}, parallel=NS()),
                  hardware=NS(get_component=lambda _name:NS(normalized_kind="hbm")))
    runtime = NS(plan=NS(scenario=scenario),
        _execution_resource_kernel=NS(physical_runtime=NS(runtimes={
            "unused_hbf": NS(core=NS(config=NS(protocol_mode="generic_nand")))})))
    assert _memory_kind(runtime) == "hbm"


def test_maverick_context_features_reset_only_the_chunked_attention_coordinate():
    from heterollm_sim.native_window_producer import native_context_features
    from test_native_maverick_semantics import miniature_maverick

    scenario = miniature_maverick()
    assert native_context_features(scenario, [3]) == (4, 4)
    assert native_context_features(scenario, [4]) == (5, 1)
    assert native_context_features(scenario, [7]) == (8, 4)
    # A mixed aged batch keeps every item's absolute query position; an
    # average context would miss the 4-Token fixed-block reset.
    assert native_context_features(scenario, [3, 4]) == (9, 5)


@pytest.mark.parametrize("prompt", [8191, 8192, 8193])
def test_8192_chunk_feature_matches_the_real_native_first_query_ir(prompt):
    from dataclasses import replace
    from heterollm_sim.ir import ModelSpec, build_model_graph_from_layer_specs
    from heterollm_sim.planner import compile_serving_cohort_schedule
    from heterollm_sim.serving import BatchCohort, BatchItem
    from heterollm_sim.native_window_producer import native_context_features
    from test_native_maverick_semantics import miniature_maverick

    scenario = miniature_maverick(prompt=prompt)
    layers = [replace(instance.layer, attention_chunk_size=8192)
              if instance.layer.attention_chunk_size else instance.layer
              for instance in scenario.model._execution_view.layer_instances]
    graph = build_model_graph_from_layer_specs("native-8192-boundary", layers,
        architecture="llama4", vocabulary_size=128, max_sequence_length=16384,
        embedding_weight_bytes=64*128*2, output_weight_bytes=64*128*2, output_head_dtype="bf16")
    scenario = replace(scenario, model=ModelSpec(name="native-8192-boundary", graph=graph))
    item = BatchItem("native-mav", "prefill", 1, prompt-1, kv_append_tokens=1,
        kv_materialized_tokens=1, main_tokens=0, draft_tokens=0, verifier_tokens=1,
        committed_tokens=0, logit_tokens=1, completion_cursor=prompt)
    schedule = compile_serving_cohort_schedule(scenario, BatchCohort("boundary", "prefill", 0., (item,)))
    reads = {}
    for row in schedule.tasks:
        contract = row.metadata.get("generic_kv_storage")
        if contract is not None:
            reads[row.metadata["layer_id"]] = contract["read_tokens"]
    full, chunk = native_context_features(scenario, [item.context_tokens])
    assert reads["dense3"] == full == prompt
    assert reads["dense0"] == chunk == (prompt-1) % 8192+1


def test_different_owners_with_a_shared_physical_command_prefix_are_not_parallel():
    from heterollm_sim.memory_types import DramConfig

    config = DramConfig(metadata={"command_resource_prefix": "shared-command"})
    tasks = []
    for rank in (0, 1):
        owner = "memory"+str(rank)
        tasks.append(task(str(rank), rank, metadata={"physical_owner": owner,
            "physical_memory_config": config,
            "memory_accesses": ({"operation": "read", "byte_count": 64,
                "physical_owner": owner, "resource_id": owner},)}))
    stage = _PreparedExecutionStage("body", 0, (), ("r0",), "gpu0", 0., tuple(tasks))
    summary = summarize_native_stages((stage,), NS())
    proof = summary["native_resource_order_proof"][0]
    assert proof["applied_phase_ns"] == proof["serial_access_phase_ns"]
    assert "@physical-prefix:shared-command" in proof["rank_private_blocks"][0]["shared_resources"]


@pytest.mark.parametrize("alias", (False, True))
def test_unallocated_native_resident_declaration_can_prove_cyclic_cold_scan_but_alias_cannot(alias):
    from heterollm_sim.memory_allocator import PhysicalAddressAllocator
    from heterollm_sim.memory_types import NandConfig
    from heterollm_sim.nand_core import NandCore

    config = NandConfig(kind="HBF", channels=2, page_bytes=4096,
        read_cache_pages_per_bank=2, capacity_bytes=1024**2)
    allocator = PhysicalAddressAllocator(config.effective_capacity_bytes, config.page_bytes)
    declaration = {"buffer_id": "weight", "generation": 0, "physical_owner": "hbf",
        "size_bytes": 6*4096, "allocation_arena": "resident"}
    extra = ({"buffer_id": "alias", "generation": 0, "physical_owner": "hbf",
        "size_bytes": 4096, "alias_of": "weight", "alias_generation": 0},) if alias else ()
    access = {"operation": "read", "physical_owner": "hbf", "buffer_id": "weight",
        "allocation_generation": 0, "offset_bytes": 0, "byte_count": 6*4096}
    row = task("scan", 0, metadata={"physical_owner": "hbf", "memory_accesses": (access,),
        "physical_allocations": (declaration, *extra), "weight_tensor_id": "weight",
        "weight_lifecycle_mode": "preloaded_resident"})
    stage = _PreparedExecutionStage("body", 0, (), ("r0",), "gpu0", 0., (row,))
    runtime = NS(allocators={"hbf": allocator}, runtimes={"hbf": NS(core=NandCore(config))})
    assert allocator.get_allocation("weight", 0) is None
    assert _cold_resident_nand_weights((stage,), runtime) == (set() if alias else {("hbf", "weight", 0)})
    assert allocator.get_allocation("weight", 0) is None


def test_refresh_guard_counts_active_io_epochs_and_their_feedback():
    from heterollm_sim.native_window_producer import _refresh_serial_bound
    from test_hbm_compiled_control import configuration

    enabled = configuration(refresh_policy="all_bank", refresh_interval_ns=100., refresh_cycle_ns=20.)
    short = _refresh_serial_bound(enabled, 32, "read")
    long = _refresh_serial_bound(enabled, 1024**2, "read")
    assert long > short > enabled.refresh_cycle_ns
    assert _refresh_serial_bound(configuration(), 1024**2, "read") == 0.


def test_nand_cache_and_rmw_phase_include_full_internal_page_transfers():
    from heterollm_sim.native_window_producer import _phase
    from heterollm_sim.memory_types import NandConfig

    config = NandConfig(channels=2, page_bytes=4096, internal_bandwidth_gb_s=1.,
                        read_cache_pages_per_bank=2, page_read_ns=10.)
    _row, _ref, cache = _phase(config, 8192, "read")
    assert cache >= 2*10.+2*2*4096.
    rmw, _ref, _cache = _phase(config, 1, "write")
    assert rmw == 2*(10.+4096.)


@pytest.mark.parametrize("case", ("hbm_tp2", "hbf_weights_kv_tp1"))
def test_future_PD_uses_own_native_chunk_geometry_and_never_executes_an_IO(case):
    from dataclasses import replace
    from test_native_external_kv import ingress_scenario, native_runtime
    from heterollm_sim.execution_control import _NativeLayerSamplingCutoff
    from heterollm_sim.native_window_producer import _native_pd_evidence

    scenario = ingress_scenario(case, warmup=3, measured=6, count=4)
    requests = list(scenario.workload.requests)
    requests[-1] = replace(requests[-1], prompt_tokens=13)
    scenario = replace(scenario, workload=replace(scenario.workload, requests=tuple(requests)))
    runtime = native_runtime(scenario)
    def stop(*args):
        raise _NativeLayerSamplingCutoff()
    runtime._native_layer_sampler = stop
    with pytest.raises(_NativeLayerSamplingCutoff):
        runtime.run_until_decode_steps(1)
    executor = runtime._external_executor
    before = (runtime.now, len(runtime.events), dict(executor.counters),
              dict(executor.runtime.timeline.ready_ns), executor._cost_phase_count())
    evidence = _native_pd_evidence(runtime)
    assert evidence["qualified"]
    after = (runtime.now, len(runtime.events), dict(executor.counters),
             dict(executor.runtime.timeline.ready_ns), executor._cost_phase_count())
    assert before == after
    longer = next(row for row in evidence["future_requests"] if row["request_id"] == "r3")
    assert longer["observed_bulk_shape_expansion"] > 1.
    assert longer["initial_state_phase_serial_upper_ns"] > 0.
    assert any(row["component_id"] == "hostmem0" and row["operation"] == "read"
               for row in longer["memory_phases"])
    assert any(row["operation"] == "write" for row in longer["memory_phases"])
    assert evidence["future_PD_serial_upper_ns"] == sum(row["native_import_upper_ns"]
        for row in evidence["future_requests"])


@pytest.mark.parametrize("pages", (1, 3, 7, 15, 20))
@pytest.mark.parametrize("shared_arrays", (False, True))
def test_native_cache_release_guard_contains_real_cold_to_warm_service_change(pages, shared_arrays):
    from heterollm_sim.data_motion import PhysicalRuntimeContext
    from heterollm_sim.memory_types import AccessRequest, NandConfig, Operation
    from heterollm_sim.native_window_producer import _nand_cache_release_phase

    config = NandConfig(kind="HBF", channels=2, planes_per_lun=2,
        planes_independent=not shared_arrays, page_bytes=4096,
        read_cache_pages_per_bank=2, page_read_ns=1000., internal_bandwidth_gb_s=1.)
    runtime = PhysicalRuntimeContext(capture_details=False)
    core = runtime.runtime(config, "hbf").core
    first = core.submit(AccessRequest("cold", Operation.READ, 0, pages*4096))
    second = core.submit(AccessRequest("warm", Operation.READ, 0, pages*4096,
        arrival_ns=first.completion_ns+1.))
    bound = _nand_cache_release_phase(core.config, pages*4096, runtime)
    first_duration = first.completion_ns
    second_duration = second.completion_ns-first.completion_ns-1.
    assert abs(first_duration-second_duration) <= bound+1e-7


def test_native_cache_release_does_not_assume_aliased_channels_are_parallel():
    from heterollm_sim.data_motion import PhysicalRuntimeContext
    from heterollm_sim.memory_types import NandConfig
    from heterollm_sim.native_window_producer import _nand_cache_release_phase

    runtime = PhysicalRuntimeContext(capture_details=False)
    config = NandConfig(channels=2, read_cache_pages_per_bank=2)
    runtime.timeline.resource_aliases["nand:channel:0"] = "shared-wire"
    runtime.timeline.resource_aliases["nand:channel:1"] = "shared-wire"
    assert _nand_cache_release_phase(config, 4096, runtime) is None


@pytest.mark.parametrize("row_bytes", (128, 256, 512))
def test_fixed_native_KV_row_keeps_real_RMW_programs_but_has_no_state_difference(row_bytes):
    from heterollm_sim.data_motion import PhysicalRuntimeContext
    from heterollm_sim.memory_types import AccessRequest, NandConfig
    from heterollm_sim.native_window_producer import _native_fixed_kv_partial_write

    config = NandConfig(kind="HBF", channels=2, page_bytes=4096, read_cache_pages_per_bank=2)
    runtime = PhysicalRuntimeContext(config, "hbf", capture_details=False)
    allocator = runtime.allocators["hbf"]
    identity = "kv:r0:L0:rank0:k"
    allocation = allocator.allocate(identity, 64*row_bytes, allocation_arena="resident")
    core = runtime.runtimes["hbf"].core
    now = 0.
    for position in (0, 1, 4096//row_bytes-1, 4096//row_bytes, 63):
        access = {"operation": "write", "physical_owner": "hbf", "buffer_id": identity,
            "byte_count": row_bytes, "offset_bytes": position*row_bytes, "allocation_generation": 0}
        contract = {"schema": "analytical-token-major-kv/v1", "row_bytes": row_bytes,
            "layout": "separate_token_major_k_and_v", "materialized_tokens": 1,
            "owners": [{"id": identity[:-2], "write_positions": (position,)}]}
        metadata = {"generic_kv_storage": contract, "physical_memory_config": core.config,
            "memory_accesses": (access,), "physical_owner": "hbf"}
        assert _native_fixed_kv_partial_write(core.config, access, metadata, runtime)
        result = core.submit(AccessRequest(str(position), "write",
            allocation.base_address+position*row_bytes, row_bytes, arrival_ns=now))
        now = result.completion_ns
        assert result.counters["rmw_read_operations"] == result.counters["pages_programmed"] == 1
        row = task("fixed-row", 0, metadata=metadata)
        stage = _PreparedExecutionStage("body", 0, (), ("r0",), "gpu0", 0., (row,))
        summary = summarize_native_stages((stage,), NS(), runtime)
        assert summary["owner_accesses"]["hbf"]["row_phase_ns"] == 0.
        assert summary["owner_accesses"]["hbf"]["fixed_kv_rmw_bulk_access_count"] == 1
        assert not _native_fixed_kv_partial_write(core.config, {**access, "offset_bytes": access["offset_bytes"]+1},
                                                 metadata, runtime)
        assert not _native_fixed_kv_partial_write(core.config, access,
            {**metadata, "generic_kv_storage": {**contract, "materialized_tokens": 2}}, runtime)
    allocator.allocate("alias", row_bytes, alias_of=identity, alias_generation=0)
    assert not _native_fixed_kv_partial_write(core.config, access, metadata, runtime)


def test_fixed_native_KV_proof_qualifies_each_real_owner_in_a_multi_request_batch():
    from heterollm_sim.data_motion import PhysicalRuntimeContext
    from heterollm_sim.memory_types import NandConfig
    from heterollm_sim.native_window_producer import _native_fixed_kv_partial_write

    runtime = PhysicalRuntimeContext(NandConfig(kind="HBF", channels=2), "hbf", capture_details=False)
    owners = [{"id": "kv:r"+str(index)+":L0:rank0", "write_positions": (17,)} for index in range(8)]
    contract = {"schema": "analytical-token-major-kv/v1", "row_bytes": 256,
        "layout": "separate_token_major_k_and_v", "materialized_tokens": 8, "owners": owners}
    for owner in owners:
        for part in ("k", "v"):
            identity = owner["id"]+":"+part
            runtime.allocators["hbf"].allocate(identity, 64*256, allocation_arena="resident")
            access = {"operation": "write", "physical_owner": "hbf", "buffer_id": identity,
                "byte_count": 256, "offset_bytes": 17*256, "allocation_generation": 0}
            assert _native_fixed_kv_partial_write(runtime.runtimes["hbf"].core.config, access,
                                                  {"generic_kv_storage": contract}, runtime)
            assert not _native_fixed_kv_partial_write(runtime.runtimes["hbf"].core.config,
                {**access, "offset_bytes": 18*256}, {"generic_kv_storage": contract}, runtime)


@pytest.mark.parametrize("remote", (False, True))
def test_real_KV_store_retains_native_contract_and_rebinds_its_integer_row(monkeypatch, remote):
    """Exercise native mapper/lowerer only; no physical request is executed."""
    from dataclasses import replace
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from exploring_hbf_native_config import build_native_comparison_scenario
    from exploring_hbf_native_pilot import build_pilot_scenario
    from heterollm_sim.control_plane_planner import plan_runtime_placement
    from heterollm_sim.memory_types import NandConfig
    from heterollm_sim.native_window_producer import _native_fixed_kv_partial_write
    from heterollm_sim.planner import TopologyAwareBatchCostProvider, native_kv_ingress_inventory
    from heterollm_sim.serving import BatchItem, BatchCohort, _OnlineRuntime, _lower_cost, compile_serving_plan

    authored = build_native_comparison_scenario("llama3_405b", "HBF", "SHORT", tp=2,
        batch_size=2, model_override=build_pilot_scenario("hbm_tp1").model,
        prompt_override=128, output_override=8, warmup_override=1, measurement_override=16)
    decision = plan_runtime_placement(authored)
    assert decision.fully_placed
    scenario = decision.apply(authored)
    if remote:
        scenario = replace(scenario, placement=replace(scenario.placement,
            metadata={**scenario.placement.metadata, "llama_backend_memory": {}}),
            hardware=replace(scenario.hardware, components=tuple(replace(component,
                metadata={**component.metadata, "memory_access_offset_bytes": 0})
                if component.is_storage else component for component in scenario.hardware.components)))
    plan = compile_serving_plan(scenario)
    lowerer = TopologyAwareBatchCostProvider(scenario, defer_physical_execution=True)
    runtime = _OnlineRuntime(plan, lowerer)
    physical = runtime._execution_resource_kernel.physical_runtime
    for step in (0, 1, 0):
        contexts = {request.request_id: request.prompt_tokens-1+step for request in plan.requests[:2]}
        items = tuple(BatchItem(request.request_id, "decode", 1, contexts[request.request_id],
            kv_append_tokens=1, kv_materialized_tokens=1, main_tokens=1, draft_tokens=0,
            verifier_tokens=1, committed_tokens=1, logit_tokens=1, completion_cursor=contexts[request.request_id]+1)
            for request in plan.requests[:2])
        cohort = BatchCohort("native-store-"+str(step), "decode", 0., items)
        cost = _lower_cost(lowerer, scenario, cohort)
        stages, fallback = runtime._execution_stage_metadata_cache.resolve(cost.metadata)
        assert stages and not fallback
        stages = runtime._with_gpu_controller_stages(stages, cohort)
        for stage in stages:
            for row in stage.execution_tasks:
                md = row.metadata
                for access in md.get("memory_accesses", ()):
                    owner = access.get("physical_owner", md.get("physical_owner"))
                    raw = md.get("physical_memory_configs", {}).get(owner, md.get("physical_memory_config"))
                    physical.runtime(raw, owner)
        for request in plan.requests[:2]:
            for declaration in native_kv_ingress_inventory(scenario, request.request_id, request.prompt_tokens-1)["physical_allocations"]:
                physical.allocators[declaration["physical_owner"]].register(dict(declaration))
        layer_id = scenario.model._execution_view.layer_instances[0].layer.layer_id
        subset = tuple(replace(stage, execution_tasks=tuple(row for row in stage.execution_tasks
            if row.metadata.get("layer_id") == layer_id)) for stage in stages
            if any(row.metadata.get("layer_id") == layer_id for row in stage.execution_tasks))
        stores = []
        for stage in subset:
            for row in stage.execution_tasks:
                md = row.metadata
                for access in md.get("memory_accesses", ()):
                    owner = access.get("physical_owner", md.get("physical_owner"))
                    active = physical.runtimes[owner].core
                    if isinstance(active.config, NandConfig) and access["operation"] == "write":
                        stores.append(access)
                        assert md["event_kind"] == ("kv_store" if remote else "kv_materialize")
                        contract = md["generic_kv_storage"]
                        owner_contract, = (row for row in contract["owners"]
                            if access["buffer_id"] in (row["id"]+":k", row["id"]+":v"))
                        assert access["offset_bytes"] == contexts[owner_contract["request_id"]]*contract["row_bytes"]
                        assert _native_fixed_kv_partial_write(active.config, access, md, physical)
        assert len(stores) == 8
        summary = summarize_native_stages(subset, scenario, physical)
        assert sum(row.get("fixed_kv_rmw_bulk_access_count", 0) for row in summary["owner_accesses"].values()) == 8
    assert all(not any(lanes) for lanes in physical.timeline.lane_available.values())
    assert not physical.timeline.busy_ns and not physical.timeline.bytes_moved
    assert all(active.clock_ns == 0. for active in physical.runtimes.values())


@pytest.mark.parametrize("blocked", ("none", "no_dependency", "duplicate_read", "alias", "raw_read"))
def test_dominating_weight_scan_proves_unique_KV_cold_and_preserves_physical_misses(blocked):
    from heterollm_sim.data_motion import PhysicalRuntimeContext
    from heterollm_sim.memory_types import AccessRequest, NandConfig
    from heterollm_sim.native_window_producer import _native_swept_kv_reads

    config = NandConfig(kind="HBF", channels=2, page_bytes=4096, read_cache_pages_per_bank=2)
    runtime = PhysicalRuntimeContext(config, "hbf", capture_details=False)
    allocator, core = runtime.allocators["hbf"], runtime.runtimes["hbf"].core
    weight = allocator.allocate("tensor:weight", 6*4096, allocation_arena="resident")
    identity = "kv:r0:L0:rank0:k"
    kv = allocator.allocate(identity, 4*4096, allocation_arena="resident")
    scan_access = {"operation": "read", "physical_owner": "hbf", "buffer_id": weight.buffer_id,
        "allocation_generation": 0, "offset_bytes": 0, "byte_count": weight.size_bytes}
    read_access = {"operation": "read", "physical_owner": "hbf", "buffer_id": identity,
        "allocation_generation": 0, "offset_bytes": 0, "byte_count": kv.size_bytes}
    scan = task("sweep", 0, metadata={"physical_owner": "hbf", "memory_accesses": (scan_access,),
        "physical_memory_config": core.config, "weight_tensor_id": "weight",
        "weight_lifecycle_mode": "preloaded_resident"})
    read = task("KV", 0, () if blocked == "no_dependency" else ("sweep",), metadata={
        "physical_owner": "hbf", "memory_accesses": (read_access,), "physical_memory_config": core.config,
        "generic_kv_storage": {"schema": "analytical-token-major-kv/v1"}})
    rows = [scan, read]
    if blocked == "duplicate_read":
        rows.append(task("second-KV", 0, ("KV",), metadata=read.metadata))
    elif blocked == "alias":
        allocator.allocate("alias", 4096, alias_of=identity, alias_generation=0)
    elif blocked == "raw_read":
        rows.append(task("opaque-read", 0, metadata={"memory_accesses": (
            {"operation": "read", "physical_owner": "hbf", "byte_count": 4096},)}))
    stage = _PreparedExecutionStage("body", 0, (), ("r0",), "gpu0", 0., tuple(rows))
    weights = _cold_resident_nand_weights((stage,), runtime)
    proof = _native_swept_kv_reads((stage,), runtime, weights)
    assert proof == ({("hbf", identity, 0)} if blocked == "none" else set())
    if blocked == "none":
        now = 0.
        hits = []
        for index, allocation in enumerate((kv, kv, weight, kv)):
            result = core.submit(AccessRequest(str(index), "read", allocation.base_address,
                allocation.size_bytes, arrival_ns=now))
            now = result.completion_ns
            hits.append(result.counters["read_cache_hits"])
        assert hits[1] == 4 and hits[-1] == 0
        summary = summarize_native_stages((stage,), NS(), runtime)
        assert summary["dominating_resident_scan_KV_cache_proof"]["qualified_read_count"] == 1
        assert summary["owner_accesses"]["hbf"]["cache_phase_ns"] == 0.

