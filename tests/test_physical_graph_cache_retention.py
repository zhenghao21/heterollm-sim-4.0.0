"""Evict reconstructible graphs without altering physical views or execution."""
from collections import OrderedDict
from dataclasses import fields, is_dataclass, replace
import gc
import json
from pathlib import Path
import sys
import weakref

from heterollm_sim import planner as p
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.contracts import _PreparedExecutionStage, _PreparedExecutionTask
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.serving import _ExecutionStageMetadataCache


def test_physical_stage_cache_releases_full_graphs_but_keeps_nonphysical_hits():
    cache = _ExecutionStageMetadataCache()
    references = []
    for index in range(12):
        task = _PreparedExecutionTask(str(index), (), ("request",), (),
            metadata={"physical_memory_config": {"kind": "GDDR7"}, "payload": bytearray(65536)})
        stage = _PreparedExecutionStage(str(index), index, (), ("request",), "gpu0", 1, (task,))
        references.append(weakref.ref(stage))
        rows = p._PreparedExecutionStageRows((), (stage,), _token=p._PREPARED_EXECUTION_STAGE_TOKEN)
        metadata = {"execution_stage_source": "executed_task_dag_kernel_timeline", "execution_stages": rows}
        assert cache.resolve(metadata) == ((stage,), None)
        assert not cache._values
    del metadata, rows, stage, task
    gc.collect()
    assert all(reference() is None for reference in references)
    plain_task = _PreparedExecutionTask("plain", (), ("request",), ())
    plain = _PreparedExecutionStage("plain", 0, (), ("request",), "gpu0", 1, (plain_task,))
    rows = p._PreparedExecutionStageRows((), (plain,), _token=p._PREPARED_EXECUTION_STAGE_TOKEN)
    metadata = {"execution_stage_source": "executed_task_dag_kernel_timeline", "execution_stages": rows}
    first = cache.resolve(metadata)
    assert cache.resolve(metadata) is first
    assert len(cache._values) == 1


def source_scenario():
    path = Path(__file__).parents[1] / "docs/frontend_native_validation_2026-10-07/scenario_qwen3_0_6b_f16_512_128.json"
    scenario = scenario_from_dict(json.loads(path.read_text(encoding="utf-8")))
    first = scenario.workload.requests[0]
    return replace(scenario, workload=replace(scenario.workload,
        requests=tuple(replace(first, request_id=name) for name in ("warmup", "measured"))))


def timeline(scenario, tasks):
    kernel = UnifiedEventKernel.from_closed_graph(tasks,
        resource_capacities=p._scenario_resource_capacities(scenario),
        resource_owners=p._scenario_resource_owners(scenario), capture_physical_details=False)
    events = []
    while (event := kernel.step()) is not None:
        events.append((event.task.task_id, event.start_ns, event.end_ns, event.demands,
                       event.task.metadata.get("physical_execution")))
    kernel.assert_drained()
    return events, kernel.makespan_ns


def retained_bytes(value, seen=None):
    """Unique Python objects owned by a cache, excluding unrelated globals."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(retained_bytes(key, seen) + retained_bytes(item, seen) for key, item in value.items())
    elif isinstance(value, (tuple, list, set, frozenset)):
        size += sum(retained_bytes(item, seen) for item in value)
    elif is_dataclass(value) and not isinstance(value, type):
        size += sum(retained_bytes(getattr(value, field.name), seen) for field in fields(value))
    return size


def test_source_f32_request_context_keys_rebuild_identical_physical_tasks(monkeypatch):
    scenario = source_scenario()
    context = p.CompilationContext(scenario, eager_full_attention_segments=False,
        compiled_serving_invocation_segments=True)
    with p._compilation_scope(scenario, context):
        layer = p._execution_layers(scenario)[0]
        plan, router = p._parallel_plan(scenario), p._topology_router(scenario)
    # A single real GGUF layer is enough to exercise F32 intermediates,
    # transposed F16 KV ranges and the complete invocation template contract.
    monkeypatch.setattr(p, "_execution_layers", lambda _scenario: (layer,))
    monkeypatch.setattr(p, "_compile_parallel_embedding", lambda builder, scenario, plan, router,
        phase, dependencies, **kw: dependencies[0])
    monkeypatch.setattr(p, "_compile_parallel_lm_head", lambda builder, scenario, plan, router,
        phase, dependencies, **kw: dependencies[0])
    monkeypatch.setattr(p, "_add_host_visible_logits_sampling_commit", lambda builder, scenario,
        plan, router, dependencies, **kw: dependencies[0])
    monkeypatch.setattr(p, "_final_output_selection", lambda *args, **kw: None)
    stage_cache = _ExecutionStageMetadataCache()
    legacy_retained = {}

    def compile_one(owner, tokens):
        with p._compilation_scope(scenario, context):
            request = next(item for item in scenario.workload.requests if item.request_id == owner)
            builder = p._TaskBuilder(replace(request, request_id="cohort-test"))
            dependency = p._add_join(builder, "decode.input", ())
            lane = p._ServingInvocationLane(0, owner, "decode", 0, tokens, tokens - 1,
                                            1, 1, True, True, False, False)
            group = p._ServingInvocationGroup(0, "decode", (lane,), None, "serial_stateful_position")
            binding = p._serving_invocation_segment_binding(builder, scenario, plan, group, (dependency,))
            assert binding is not None
            p._compile_or_replay_serving_invocation(builder, scenario, plan, router, group,
                                                   phase="decode", dependencies=(dependency,))
            tasks = p._promote_physical_allocation_extents(builder.tasks)
            # Capture an exact, fixed-context segment to exercise eviction
            # independently of dynamic-attention admission (source-fused QK
            # scaling currently makes that broader admission return None).
            captured = p._TaskSegmentTemplate.capture(builder, first_task_index=0,
                source_prefix="decode", source_counter_before=0, source_dependencies=(),
                source_initial_previous=None, source_initial_dma=None,
                terminal_task_id=builder.tasks[-1].task_id, source_phase="decode")
            assert captured is not None
            prepared = tuple(_PreparedExecutionTask(task.task_id, task.dependencies,
                (owner,), task.demands, category=task.category, metadata=task.metadata) for task in tasks)
            stage = _PreparedExecutionStage("stage", 0, (), (owner,), "gpu0", 1, prepared)
            rows = p._PreparedExecutionStageRows((), (stage,), _token=p._PREPARED_EXECUTION_STAGE_TOKEN)
            parsed = stage_cache.resolve({"execution_stage_source": "executed_task_dag_kernel_timeline",
                                          "execution_stages": rows})
            assert parsed == ((stage,), None)
            # Exactly the ownership tuple used by resolve() before the fix.
            legacy_retained[id(rows)] = (rows, parsed, True)
            return tuple(tasks), binding[0], binding[1], captured

    exact_segments = OrderedDict()
    original, cache, first_key, first_template = compile_one("warmup", 513)
    p._task_segment_cache_put(context, exact_segments, first_key, first_template, max_entries=8)
    for tokens in range(514, 523):
        _, same_cache, key, template = compile_one("measured", tokens)
        assert same_cache is cache and key != first_key
        assert len(cache) <= 8
        p._task_segment_cache_put(context, exact_segments, key, template, max_entries=8)
        assert len(exact_segments) <= 8
    assert len(exact_segments) == 8 and first_key not in exact_segments
    rebuilt, _, key, rebuilt_template = compile_one("warmup", 513)
    assert key == first_key and rebuilt == original
    assert rebuilt_template.tasks == first_template.tasks
    rebuilt_events, rebuilt_end = timeline(scenario, rebuilt)
    original_events, original_end = timeline(scenario, original)
    assert rebuilt_events == original_events and rebuilt_end == original_end
    other, _, other_key, _ = compile_one("measured", 513)
    assert other_key != first_key
    old_ids = {row["buffer_id"] for task in original for row in task.metadata.get("memory_accesses", ())
               if row["buffer_id"].startswith("native_kv:")}
    new_ids = {row["buffer_id"] for task in other for row in task.metadata.get("memory_accesses", ())
               if row["buffer_id"].startswith("native_kv:")}
    assert old_ids and new_ids and old_ids.isdisjoint(new_ids)
    before, after = retained_bytes(legacy_retained), retained_bytes(stage_cache._values)
    assert len(legacy_retained) == 12 and not stage_cache._values
    assert before > after
    print(json.dumps({"short_scenario": "12 one-layer source-F32 cohorts",
        "legacy_retained_stage_graphs": len(legacy_retained), "new_retained_stage_graphs": 0,
        "legacy_cache_owned_python_bytes": before, "new_cache_owned_python_bytes": after,
        "retained_invocation_templates": len(exact_segments),
        "task_count": len(original), "timeline_event_count": len(original_events),
        "absolute_timeline_difference_ns": abs(rebuilt_end - original_end)}))


def test_invocation_capacity_override_does_not_reduce_other_leaf_segment_caches():
    class Context:
        leaf_cache_entries = 16
    large, leaf = OrderedDict(), OrderedDict()
    for key in range(12):
        p._task_segment_cache_put(Context(), large, key, key, max_entries=8)
        p._task_segment_cache_put(Context(), leaf, key, key)
    assert tuple(large) == tuple(range(4, 12))
    assert tuple(leaf) == tuple(range(12))
