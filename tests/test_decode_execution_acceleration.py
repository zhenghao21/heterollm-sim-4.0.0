"""Decode optimizations preserve physical execution and input ownership."""
from dataclasses import fields, is_dataclass, replace

import pytest

from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_dram_core_semantics import config


def exact_values(left, right, seen=None):
    """Include scalar types, float bits and mapping order in lowering checks."""
    if seen is None:
        seen = set()
    assert type(left) is type(right)
    pair = (id(left), id(right))
    if pair in seen:
        return
    seen.add(pair)
    if isinstance(left, float):
        assert left.hex() == right.hex()
    elif is_dataclass(left):
        for field in fields(left):
            exact_values(getattr(left, field.name), getattr(right, field.name), seen)
    elif isinstance(left, dict):
        assert tuple(left) == tuple(right)
        for key in left:
            exact_values(left[key], right[key], seen)
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            exact_values(first, second, seen)
    else:
        assert left == right


@pytest.mark.parametrize("bursts", [4, 5, 16, 63])
def test_small_host_accesses_match_every_interpreted_transition(bursts):
    configured = config(channels=2, banks_per_group=4, bank_groups_per_rank=4,
        rows_per_bank=128, row_bytes=8192, interleave_bytes=64,
        read_latency_ns=11.3, write_latency_ns=8.7, burst_interval_ns=.7,
        interface_bandwidth_gb_s=44.8)
    old = DramCore(configured, capture_details=False)
    old._can_accelerate = lambda: False
    new = DramCore(configured, capture_details=False)
    for index in range(18):
        request = AccessRequest(str(index), "write" if index % 3 == 0 else "read",
            (index % 4) * 262144 + 4093, bursts * 64 - 1, index * 133.7)
        expected, actual = old.submit(request), new.submit(request)
        assert actual.counters["compiled_float64"] is True
        assert replace(actual, counters={}) == replace(expected, counters={})
        assert {key: actual.counters[key] for key in expected.counters} == expected.counters
        assert old._banks == new._banks
        for field in ("ready_ns", "busy_ns", "bytes_moved", "last_intervals", "directions"):
            assert getattr(old.timeline, field) == getattr(new.timeline, field)
        assert old._inflight == new._inflight


def test_lowering_reuses_only_its_bound_immutable_component_config(monkeypatch):
    from heterollm_sim import physical_contract, planner
    from test_hbf_host_staging_integration import strict_staging_scenario

    scenario = strict_staging_scenario(prompt=4, output=2, all_hbf=True, capacity=65536)
    component = next(item for item in scenario.hardware.components if item.normalized_kind == "hbf")
    calls = []
    validate = physical_contract.require_physical_memory_config

    def counted(item):
        calls.append(item)
        return validate(item)

    monkeypatch.setattr(physical_contract, "require_physical_memory_config", counted)
    context = planner.CompilationContext(scenario)
    with planner._compilation_scope(scenario, context):
        first = planner._cached_physical_memory_config(component)
        assert planner._cached_physical_memory_config(component) is first
        other = replace(component, metadata=dict(component.metadata))
        assert planner._cached_physical_memory_config(other) == first
    outside = planner._cached_physical_memory_config(component)
    assert outside == first and outside is not first
    assert len(calls) == 3


@pytest.mark.parametrize("medium", ["HBM", "HBF"])
def test_physical_dense_segment_rebinds_every_request_generation(medium, monkeypatch):
    from heterollm_sim import planner
    from test_hbf_rope_dataflow import build_scan_scenario

    scenario = build_scan_scenario("qwen2_5-0_5b", medium, 32, 3)
    context = planner.CompilationContext(scenario)
    with planner._compilation_scope(scenario, context):
        plan, router = context.parallel_plan(), context.router()
        layer = planner._execution_layers(scenario)[0]
        def emit(request_id, use_cache):
            request = replace(scenario.workload.requests[0], request_id=request_id)
            builder = planner._TaskBuilder(request)
            start = builder.add("input", planner.TaskCategory.COMPUTE)
            builder._rank_value_components[start] = {0: "gpu0"}
            fn = (planner._compile_parallel_dense_mlp if use_cache else
                  planner._compile_parallel_dense_mlp_uncached)
            fn(builder, scenario, plan, router, layer, 1, request_id + ".decode.layer-000", (start,))
            return tuple(builder.tasks)
        emit("request-first", True)
        cache = context.invariant(("dense_mlp_task_segment_cache_v1",), dict)
        assert cache and all(template.physical_bindings is not None for template in cache.values())
        attached = []
        attach = planner._attach_gddr_physical_task
        def counted(task, configured):
            attached.append(task.task_id)
            return attach(task, configured)
        monkeypatch.setattr(planner, "_attach_gddr_physical_task", counted)
        actual = emit("request-second", True)
        # The input task is outside the cached fragment. Its complete static
        # MLP binds slots without rebuilding any physical descriptor contract.
        assert len(attached) == 1
        expected = emit("request-second", False)
        assert actual == expected
        exact_values(actual, expected)
        assert any(task.metadata.get("memory_accesses") for task in actual)
        for new, old in zip(actual, expected, strict=True):
            if "physical_memory_config" in new.metadata:
                assert new.metadata["physical_memory_config"] is old.metadata["physical_memory_config"]
            for owner, configured in new.metadata.get("physical_memory_configs", {}).items():
                assert configured is old.metadata["physical_memory_configs"][owner]
            for owner, contract in new.metadata.get("hbf_host_staging_by_owner", {}).items():
                old_contract = old.metadata["hbf_host_staging_by_owner"][owner]
                assert contract["host_config"] is old_contract["host_config"]
                assert contract is not old_contract
                assert contract["routes"] is not old_contract["routes"]
                for direction, hops in contract["routes"].items():
                    for hop, old_hop in zip(hops, old_contract["routes"][direction], strict=True):
                        assert hop is not old_hop
                        assert hop["resource_capacities"] is not old_hop["resource_capacities"]
        physical = next(task for task in actual if task.metadata.get("memory_accesses"))
        physical.metadata["memory_accesses"][0]["byte_count"] = 1
        if medium == "HBF":
            contract = next(iter(physical.metadata["hbf_host_staging_by_owner"].values()))
            hops = next(hops for hops in contract["routes"].values() if hops)
            hops[0]["latency_ns"] = 999999.0
        # Published task metadata cannot mutate the resident template or the
        # independently bound routes/descriptors of the following request.
        assert emit("request-third", True) == emit("request-third", False)


@pytest.mark.parametrize("prompt,output,cpu_rows", [(64, 3, 1), (32, 7, 1), (32, 3, 2)])
def test_dense_segment_keeps_request_lengths_and_host_graph_rows(prompt, output, cpu_rows, monkeypatch):
    from heterollm_sim import planner
    from test_hbf_rope_dataflow import build_scan_scenario

    scenario = build_scan_scenario("qwen2_5-0_5b", "HBM", 32, 3)
    context = planner.CompilationContext(scenario)
    original = planner._compile_parallel_dense_mlp_uncached
    calls = []
    def counted(*args):
        calls.append(args[0].request.request_id)
        return original(*args)
    monkeypatch.setattr(planner, "_compile_parallel_dense_mlp_uncached", counted)
    with planner._compilation_scope(scenario, context):
        plan, router = context.parallel_plan(), context.router()
        layer = context.execution_layers()[0]
        def emit(request_id, prompt_tokens, output_tokens, rows, cached):
            request = replace(scenario.workload.requests[0], request_id=request_id,
                              prompt_tokens=prompt_tokens, output_tokens=output_tokens)
            builder = planner._TaskBuilder(request)
            builder._cpu_graph_rows = rows
            start = builder.add("input", planner.TaskCategory.COMPUTE)
            builder._rank_value_components[start] = {0: "gpu0"}
            fn = planner._compile_parallel_dense_mlp if cached else original
            fn(builder, scenario, plan, router, layer, 1,
               request_id + ".decode.layer-000", (start,))
            return tuple(builder.tasks)
        emit("first", 32, 3, 1, True)
        actual = emit("second", prompt, output, cpu_rows, True)
        # Host graph rows select CPU thread profiles independently of the
        # one-row operator shape; their change requires fresh lowering too.
        assert calls == ["first", "second"]
        assert actual == emit("second", prompt, output, cpu_rows, False)
        replay = emit("third", prompt, output, cpu_rows, True)
        assert calls == ["first", "second"]
        assert replay == emit("third", prompt, output, cpu_rows, False)


def test_dense_segment_temporary_same_id_layer_bypasses_bound_cache():
    from heterollm_sim import planner
    from test_hbf_rope_dataflow import build_scan_scenario

    scenario = build_scan_scenario("qwen2_5-0_5b", "HBM", 32, 3)
    context = planner.CompilationContext(scenario)
    with planner._compilation_scope(scenario, context):
        plan, router = context.parallel_plan(), context.router()
        layer = context.execution_layers()[0]
        temporary = replace(layer, intermediate_size=layer.intermediate_size + 64)
        def emit(selected, cached):
            builder = planner._TaskBuilder(scenario.workload.requests[0])
            start = builder.add("input", planner.TaskCategory.COMPUTE)
            builder._rank_value_components[start] = {0: "gpu0"}
            fn = (planner._compile_parallel_dense_mlp if cached else
                  planner._compile_parallel_dense_mlp_uncached)
            fn(builder, scenario, plan, router, selected, 1, "decode.layer-000", (start,))
            return tuple(builder.tasks)
        bound_tasks = emit(layer, True)
        cache = context.invariant(("dense_mlp_task_segment_cache_v1",), dict)
        saved_templates = tuple(cache.items())
        actual = emit(temporary, True)
        assert actual == emit(temporary, False)
        assert actual != bound_tasks
        assert tuple(cache.items()) == saved_templates


def test_summary_publishes_exact_multilane_and_unknown_interval_completions():
    from types import SimpleNamespace
    from heterollm_sim.event_kernel import _publish_physical_summary

    prior = {"task_id": "before", "resource_id": "single", "start_ns": 1., "end_ns": 4.}
    kernel = SimpleNamespace(
        physical_runtime=SimpleNamespace(timeline=SimpleNamespace(lane_available={
            "single": [9.], "multiple": [8., 11.], "unknown": [7.], "unchanged": [3.]})),
        resource_last_interval={"single": prior}, _resource_lane_last_interval={},
        resource_available={})
    task = SimpleNamespace(task_id="now", metadata={"physical_resource_last_intervals": {
        "single": (4., 9.), "multiple": (6., 11.)}})
    before = {"timeline": {"lane_available": {
        "single": [4.], "multiple": [5., 6.], "unknown": [2.], "unchanged": [3.]}}}
    predecessors = {}
    result = _publish_physical_summary(kernel, task,
        ("single", "multiple", "unknown", "unchanged"), before, predecessors)
    assert result == (("single", 0, 9.), ("multiple", 0, 8.), ("multiple", 1, 11.), ("unknown", 0, 7.))
    assert predecessors == {"single": prior}
    assert kernel.resource_available == {"single": 9., "multiple": 8., "unknown": 7., "unchanged": 3.}
    assert kernel._resource_lane_last_interval[("multiple", 1)] == {
        "task_id": "now", "resource_id": "multiple", "start_ns": 6., "end_ns": 11., "lane": 1}
    assert "start_ns" not in kernel._resource_lane_last_interval[("multiple", 0)]
    assert "start_ns" not in kernel._resource_lane_last_interval[("unknown", 0)]


@pytest.mark.parametrize("capture_details", [False, True])
def test_zero_duration_physical_reservation_publishes_advanced_lane(capture_details):
    from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
    from heterollm_sim.event_kernel import UnifiedEventKernel
    from heterollm_sim.memory_types import make_hbf_config

    configured = make_hbf_config(capacity_bytes=65536, page_bytes=4096,
                                 page_read_ns=0., front_ns=10.)
    kernel = UnifiedEventKernel(capture_physical_details=capture_details)
    kernel.submit([TaskSpec("physical", "request", "physical", TaskCategory.MEMORY,
        metadata={"physical_memory_config": configured, "physical_owner": "memory",
                  "memory_accesses": [{"operation": "read", "address": 0,
                                       "byte_count": 4096}]})])
    event = kernel.step()
    resource = "memory:nand:array:0"
    assert resource not in event.task.metadata["physical_demands_resource_ids"]
    assert kernel.physical_runtime.timeline.lane_available[resource] == [10.]
    assert kernel.resource_available[resource] == 10.
    kernel.submit([TaskSpec("after", "request", "after", TaskCategory.COMPUTE,
                           demands=(ResourceDemand(resource, 1.),))])
    after = kernel.step()
    assert (after.start_ns, after.end_ns) == (10., 11.)
    assert kernel.physical_runtime.timeline.lane_available[resource] == [11.]


@pytest.mark.parametrize("medium", ["dram", "nand"])
def test_core_metrics_respect_custom_timeline_writers_and_reducers(medium):
    from heterollm_sim.memory_transfer import ResourceTimeline
    from heterollm_sim.memory_types import make_hbf_config
    from heterollm_sim.nand_core import NandCore

    class InstrumentedTimeline(ResourceTimeline):
        def reserve(self, *args, **kwargs):
            result = super().reserve(*args, **kwargs)
            self.busy_ns["instrument"] = self.busy_ns.get("instrument", 0.) + 1.
            return result

        def metrics_delta(self, before):
            return {**super().metrics_delta(before), "custom_observation": True}

    timeline = InstrumentedTimeline(busy_ns={f"idle-{index}": 0. for index in range(30)})
    core = (DramCore(config(), timeline, capture_details=True) if medium == "dram" else
            NandCore(make_hbf_config(capacity_bytes=65536, page_bytes=4096),
                     timeline, capture_details=True))
    result = core.submit(AccessRequest("read", "read", 0, 64))
    assert timeline.busy_ns["instrument"] > 0.
    assert result.counters["resource_busy_ns"]["instrument"] == timeline.busy_ns["instrument"]
    assert result.counters["custom_observation"] is True


def test_integer_busy_counters_keep_the_original_float_accumulator(monkeypatch):
    from heterollm_sim.contracts import TaskCategory, TaskSpec
    from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
    from heterollm_sim.memory_types import TransactionResult, make_hbf_config
    from heterollm_sim.nand_core import NandCore

    busy_values = [2**53 + 1, 1]
    remaining = iter(busy_values)
    def submit(core, request):
        return TransactionResult(request.request_id, request.operation, request.arrival_ns,
            request.arrival_ns + 1., 4096, 4096, counters={
                "resource_busy_ns": {"wire": next(remaining)},
                "resource_bytes": {"wire": 4096}})
    monkeypatch.setattr(NandCore, "submit", submit)
    configured = make_hbf_config(capacity_bytes=65536, page_bytes=4096)
    task = TaskSpec("physical", "request", "physical", TaskCategory.MEMORY,
        metadata={"physical_memory_config": configured, "physical_owner": "memory",
                  "memory_accesses": [{"operation": "read", "address": index * 4096,
                                       "byte_count": 4096} for index in range(2)]})
    result = resolve_physical_task(task, PhysicalRuntimeContext(), 0.)
    expected = 0.0
    for value in busy_values:
        expected += value
    assert expected != float(sum(busy_values))
    assert result.demands[0].service_ns == expected
    row = result.metadata["physical_execution_by_owner"]["memory"]
    assert row["resource_busy_ns"]["wire"] == expected
    assert isinstance(row["resource_busy_ns"]["wire"], float)


def test_sparse_core_accounting_rereads_public_values_and_dictionary_order():
    from heterollm_sim.memory_transfer import ResourceTimeline

    timeline = ResourceTimeline(
        busy_ns={f"r{index}": float(index) for index in range(64)},
        bytes_moved={f"r{index}": index for index in range(64)})
    before = timeline.metrics_snapshot()
    timeline._touched = {"r1", "r2", "r3"}
    timeline.last_intervals = {key: (0., 1.) for key in timeline._touched}
    for key in timeline._touched:
        timeline.busy_ns[key] += 1.
        timeline.bytes_moved[key] += 64
    for iteration in range(3):
        if iteration:
            # Same-size reorder and replacement must both invalidate ordinals.
            for values in (timeline.busy_ns, timeline.bytes_moved):
                value = values.pop("r1")
                values["r1"] = value + 1
            if iteration == 2:
                timeline.busy_ns = dict(reversed(tuple(timeline.busy_ns.items())))
        expected = timeline.metrics_delta(before)
        actual = timeline._core_metrics_delta(before)
        assert actual == expected
        for field in ("resource_busy_ns", "resource_bytes"):
            assert list(actual[field]) == list(expected[field])


def test_sparse_core_accounting_respects_custom_dictionary_items_order():
    from heterollm_sim.memory_transfer import ResourceTimeline

    class ReverseItems(dict):
        def items(self):
            return reversed(tuple(super().items()))

    timeline = ResourceTimeline(busy_ns=ReverseItems(
        (f"r{index}", float(index)) for index in range(64)))
    before = timeline.metrics_snapshot()
    timeline._touched = {"r1", "r2"}
    timeline.last_intervals = {key: (0., 1.) for key in timeline._touched}
    for key in timeline._touched:
        timeline.busy_ns[key] += 1.
    actual = timeline._core_metrics_delta(before)
    assert actual == timeline.metrics_delta(before)
    assert list(actual["resource_busy_ns"]) == ["r2", "r1"]


def test_precision_cache_requires_bound_layer_identity_and_explicit_metadata_stays_live(monkeypatch):
    from heterollm_sim import planner
    from test_hbf_rope_dataflow import build_scan_scenario

    scenario = build_scan_scenario("qwen2_5-0_5b", "HBF", 32, 3)
    context = planner.CompilationContext(scenario)
    layer = context.execution_layers()[0]
    temporary = replace(layer)
    calls = []
    original = planner._layer_runtime_precision_bits
    def counted(*args):
        calls.append(args[0])
        return original(*args)
    monkeypatch.setattr(planner, "_layer_runtime_precision_bits", counted)
    with planner._compilation_scope(scenario, context):
        assert planner._activation_storage_bits(layer, scenario) == 16
        assert planner._activation_storage_bits(layer, scenario) == 16
        assert planner._layer_precision_bits(layer)[0] == 16
        assert len(calls) == 1
        planner._layer_precision_bits(temporary)
        planner._layer_precision_bits(temporary)
        planner._layer_precision_bits(layer, {})
        planner._layer_precision_bits(layer, {})
        assert len(calls) == 5
    planner._layer_precision_bits(layer)
    planner._layer_precision_bits(layer)
    assert len(calls) == 7
    f32 = replace(scenario, workload=replace(scenario.workload,
        metadata={**scenario.workload.metadata, "llama_cpp_f32_hidden_storage": True}))
    f32_context = planner.CompilationContext(f32)
    with planner._compilation_scope(f32, f32_context):
        f32_layer = f32_context.execution_layers()[0]
        assert planner._activation_storage_bits(f32_layer, f32) == 32
        assert planner._activation_storage_bits(f32_layer) == 16
        assert planner._activation_storage_bits(f32_layer, scenario) == 16


@pytest.mark.parametrize("medium", ["HBM", "HBF"])
@pytest.mark.parametrize("paged", [False, True])
def test_static_projection_replays_raw_sources_while_kv_context_pages_and_generations_change(
    medium, paged, monkeypatch
):
    from types import SimpleNamespace
    from heterollm_sim import planner
    from test_hbf_rope_dataflow import build_scan_scenario

    scenario = build_scan_scenario("qwen2_5-0_5b", medium, 32, 3)
    context = planner.CompilationContext(scenario)
    with planner._compilation_scope(scenario, context):
        plan, router = context.parallel_plan(), context.router()
        layer = context.execution_layers()[0]
        tokens_per_page = 16
        page_bytes = sum(2 * planner._kv_tensor_bytes(scenario, item, 1, 1) * tokens_per_page
                         for item in context.execution_layers())
        def emit(cohort, history, generation, cpu_rows=1):
            builder = planner._TaskBuilder(replace(scenario.workload.requests[0],
                request_id=cohort, prompt_tokens=1, output_tokens=0))
            builder._cpu_graph_rows = cpu_rows
            builder._attention_invocation_lanes = (SimpleNamespace(request_id="scan-request",
                context_tokens=history + 1, kv_read_tokens=history, kv_materialized_tokens=1),)
            if paged:
                component = "hbf0" if medium == "HBF" else "hbm0"
                builder._physical_kv_page_table = {"tokens_per_page": tokens_per_page,
                    "requests": {"scan-request": tuple({"buffer_id": f"kv-page-{index}",
                        "allocation_generation": generation, "size_bytes": page_bytes,
                        "component_id": component, "physical_owner": "memory:" + component}
                        for index in range(3))}}
            first = builder.add("input", planner.TaskCategory.COMPUTE)
            builder._rank_value_components[first] = {0: "gpu0"}
            planner._compile_parallel_layer_body(builder, scenario, plan, router, layer,
                token_batch=1, context_tokens=history, kv_read_tokens=history,
                kv_append_tokens=1, kv_materialized_tokens=1, linear_state_runtime=None,
                phase=cohort + ".decode", dependencies=(first,))
            return tuple(builder.tasks)
        first = emit("cohort-first", 15, 4)
        cache = context.invariant(("static_input_projection_segments",), dict)
        assert len(cache) == 1
        template = next(iter(cache.values()))[0]
        hits = []
        replay = planner._TaskSegmentTemplate.replay
        def counted(self, *args, **kwargs):
            if self is template:
                hits.append(1)
            return replay(self, *args, **kwargs)
        monkeypatch.setattr(planner._TaskSegmentTemplate, "replay", counted)
        actual = emit("cohort-second", 16, 7)
        assert hits == [1]
        changed_rows = emit("cohort-third", 17, 8, cpu_rows=2)
        assert len(cache) == 2
        assert hits == [1]
        monkeypatch.setattr(planner, "_compile_static_input_projection",
                            lambda *args, **kwargs: args[-1]())
        expected = emit("cohort-second", 16, 7)
        assert actual == expected
        exact_values(actual, expected)
        assert changed_rows == emit("cohort-third", 17, 8, cpu_rows=2)
        first_views = [task.metadata["generic_kv_storage"] for task in first
                       if "generic_kv_storage" in task.metadata]
        next_views = [task.metadata["generic_kv_storage"] for task in actual
                      if "generic_kv_storage" in task.metadata]
        assert first_views and next_views and first_views != next_views
        if paged:
            assert all(page["allocation_generation"] == 7 for view in next_views
                       for owner in view["owners"] for page in owner["pages"])


def test_static_projection_does_not_cache_packed_conversion_side_effects():
    from types import SimpleNamespace
    from heterollm_sim import planner
    from test_hbf_rope_dataflow import build_scan_scenario

    scenario = build_scan_scenario("qwen2_5-0_5b", "HBF", 32, 3)
    context = planner.CompilationContext(scenario)
    with planner._compilation_scope(scenario, context):
        plan = context.parallel_plan()
        rank = plan.ranks[0]
        layer = context.execution_layers()[0]
        builder = planner._TaskBuilder(scenario.workload.requests[0])
        calls = []
        def emit():
            calls.append(1)
            terminal = builder.add("projection", planner.TaskCategory.COMPUTE)
            builder._mmvq_conversion_producers[("input",)] = terminal
            return (terminal, None, None, None, None, None, None,
                    SimpleNamespace(packed_weight_formats=("Q8_0",)))
        for _ in range(2):
            planner._compile_static_input_projection(builder, scenario, plan, rank,
                layer, 1, 1, 1, "decode", "decode.layer", (), emit, eligible=True)
        assert calls == [1, 1]
        assert not context.invariant(("static_input_projection_segments",), dict)

