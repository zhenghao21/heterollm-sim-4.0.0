"""GPU frontend ownership on tiny synthetic graphs; no native/GPU measurements."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from heterollm_sim import planner
from heterollm_sim.contracts import ResourceDemand, TaskCategory
from heterollm_sim.final_layer_output_selection import SOURCE_KEY, source_declaration
from heterollm_sim.ir import RequestSpec
from heterollm_sim.llama_scenario import apply_llama_runtime_config
from heterollm_sim.runtime_adapters import apply_llama_cuda_op_offload
from tests.model_helpers import model_from_layer_specs
from tests.test_final_layer_output_selection_planner import _cohort, _scenario
from tests.test_llama_final_norm_placement import scenario as norm_scenario
from tests.test_physical_projection_invocations import _full_layer, _descriptors
from tests.test_runtime_op_offload import host_case, contract

FRONT = planner._GPU_CONSUMER_FRONTEND_STAGE
GPU_EVENTS = {"host_cohort_h2d", "iommu_translation_batch", "dma_controller_batch",
              "host_cohort_submit", "gpu_command_processor_batch", "native_phase_boundary"}


def offload_scenario(rows, enabled=True):
    base = host_case(rows=rows, enabled=enabled)
    model = model_from_layer_specs(
        "consumer-test", (_full_layer(_descriptors()),), vocabulary_size=256,
        max_sequence_length=256, embedding_weight_bytes=4096,
        architecture="llama_decoder", metadata={
            SOURCE_KEY: source_declaration(), "final_norm_epsilon": 1e-5,
            "final_norm_weight_binding": {"name": "output_norm.weight", "shape": [128],
                "type": "F32", "n_bytes": 512, "offset": 0},
        },
    )
    base = replace(base, model=model, placement=replace(base.placement,
        model_name=model.name, op_to_component={}, tensor_to_component={}, tensor_bytes={}))
    base = apply_llama_runtime_config(base, base.llama_cpp_config)
    return apply_llama_cuda_op_offload(base, base.llama_cpp_config,
        source_contract=contract(), cuda_backend_available=True)


def lowered(case, cohort):
    with planner._compilation_scope(case):
        result = planner._lower_serving_cohort(case, cohort)
    seen = set()
    for task in result.schedule.tasks:
        assert set(task.dependencies) <= seen
        seen.add(task.task_id)
    summary = planner.execute_cost_schedule(result.schedule)
    stages, reason = planner._compact_execution_stages(
        case, summary.execution_records, result.extra_metadata["operator_invocation_groups"])
    return result, summary, stages, reason


def ancestors(tasks, task_id):
    by_id = {t.task_id: t for t in tasks}
    result, pending = set(), list(by_id[task_id].dependencies)
    while pending:
        value = pending.pop()
        if value not in result:
            result.add(value)
            pending.extend(by_id[value].dependencies)
    return result


def test_all_cpu_preserves_prepare_and_pack_without_gpu_control():
    case = norm_scenario(ngl=0, offload=False)
    result, summary, stages, reason = lowered(case, _cohort(1, 1))
    assert reason is None and stages
    assert result.extra_metadata["gpu_consumer_frontend"]["by_gpu"] == {}
    kinds = [t.metadata.get("event_kind") for t in result.schedule.tasks]
    assert "host_cohort_prepare" in kinds and "host_cohort_pack" in kinds
    assert not GPU_EVENTS.intersection(kinds)
    assert not any(t.metadata.get("orchestration_stage") == FRONT for t in result.schedule.tasks)
    assert not any(s.get("stage_role") == FRONT for s in stages)
    slower = replace(case, runtime_profile=replace(case.runtime_profile,
        gpu_controllers={key: replace(value, command_processor=replace(value.command_processor,
                        command_submission_latency_ns=1e9))
                         for key, value in case.runtime_profile.gpu_controllers.items()}))
    changed, changed_summary, changed_stages, changed_reason = lowered(slower, _cohort(1, 1))
    assert changed_reason is None
    assert changed.schedule.tasks == result.schedule.tasks
    assert changed_summary.makespan_ns == summary.makespan_ns
    assert changed_stages == stages


@pytest.mark.parametrize("rows,expected", [(1, False), (64, True)])
def test_ngl_zero_uses_actual_body_dispatch_not_layer_setting(rows, expected):
    case = offload_scenario(rows)
    result, _, stages, reason = lowered(case, _cohort(rows, 1))
    assert case.llama_cpp_config.gpu_layers == 0
    mains = [t for t in result.schedule.tasks if t.metadata.get("phase") == "gpu_gemm"]
    assert bool(mains) is expected
    controls = [t for t in result.schedule.tasks if t.metadata.get("event_kind") == "host_cohort_submit"]
    assert len(controls) == int(expected)
    assert bool(result.extra_metadata["gpu_consumer_frontend"]["by_gpu"]) is expected
    assert reason is None and stages
    if expected:
        assert sum(t.metadata["submission_count"] for t in controls) == 1
        assert any(t.metadata.get("host_gemm_offload_applied") is True for t in mains)
        assert any(t.metadata.get("event_kind") == "model_weight_read" for t in result.schedule.tasks)
        scope = next(s for s in stages if s.get("stage_role") == FRONT)
        assert scope["component_id"] == "gpu0"
        assert any(scope["stage_id"] in s["dependencies"] for s in stages if s.get("group_id"))


def test_repeated_prefill_rebuilds_consumer_frontends():
    case = offload_scenario(64)
    cohort = _cohort(64, 1)
    expected = planner.compile_serving_cohort_schedule(case, cohort)
    with planner._compilation_scope(case):
        first = planner.compile_serving_cohort_schedule(case, cohort)
        second = planner.compile_serving_cohort_schedule(case, cohort)
    assert first == second == expected
    assert sum(t.metadata.get("event_kind") == "host_cohort_submit" for t in second.tasks) == 1



def test_decode_template_hit_requalifies_body_consumers():
    # This fixture uses the source-supported serial decode template contract;
    # equal-length llama runtime batches intentionally do not use that cache.
    case = _scenario("qwen3_5_hybrid_transformer")
    source = _cohort(1, 1, name="cold", context=9, phase="decode")
    target = _cohort(1, 1, name="replay", context=9, phase="decode")
    expected = planner.compile_serving_cohort_schedule(case, target)
    context = planner.CompilationContext(case, eager_full_attention_segments=False,
                                         compiled_serving_invocation_segments=True)
    with patch.object(planner, "_compile_parallel_iteration", wraps=planner._compile_parallel_iteration) as body:
        with planner._compilation_scope(case, context):
            planner.compile_serving_cohort_schedule(case, source)
            calls = body.call_count
            actual = planner.compile_serving_cohort_schedule(case, target)
    assert calls > 0 and body.call_count == calls
    assert actual == expected
    assert sum(t.metadata.get("event_kind") == "host_cohort_submit" for t in actual.tasks) == 1

def test_launch_only_is_a_consumer_but_control_resource_cannot_self_qualify():
    builder = planner._TaskBuilder(RequestSpec("unit", 0, 1, 1))
    control = builder.add("control", TaskCategory.POLICY,
        (ResourceDemand("gpu0.command_processor", 100),), metadata={"target_component": "gpu0"})
    launch = builder.add("required_zero_launch", TaskCategory.COMPUTE,
        (ResourceDemand("gpu0.frontend", 0),), metadata={"target_component": "gpu0",
         "phase": "kernel_launch", "cost_model": {"device": "gpu", "launch_only": True}})
    assert planner._actual_gpu_consumer_component(builder.tasks[0], {"gpu0"}) is None
    assert planner._actual_gpu_consumer_component(builder.tasks[1], {"gpu0"}) == "gpu0"
    with_scope = replace(builder.tasks[1], metadata={**builder.tasks[1].metadata,
                                                  "orchestration_stage": FRONT})
    assert planner._actual_gpu_consumer_component(with_scope, {"gpu0"}) is None
    unknown = replace(builder.tasks[1], metadata={"target_component": "gpu0",
        "cost_model": {"model": "gpu_custom_unqualified", "device": "gpu"}})
    assert planner._actual_gpu_consumer_component(unknown, {"gpu0"}) is None
    transfer = replace(builder.tasks[1], category=TaskCategory.COMMUNICATION,
        metadata={"target_component": "gpu0", "cost_model": {"model": "gpu_memory_roofline"}})
    assert planner._actual_gpu_consumer_component(transfer, {"gpu0"}) is None


def two_gpu_scenario():
    base = norm_scenario(ngl=0)
    gpu = next(c for c in base.hardware.components if c.component_id == "gpu0")
    links = tuple(replace(link, link_id=link.link_id + "-gpu1",
        source_component="gpu1" if link.source_component == "gpu0" else link.source_component,
        target_component="gpu1" if link.target_component == "gpu0" else link.target_component)
        for link in base.hardware.links if "gpu0" in (link.source_component, link.target_component))
    hardware = replace(base.hardware,
        components=(*base.hardware.components, replace(gpu, component_id="gpu1")),
        links=(*base.hardware.links, *links))
    runtime = replace(base.runtime_profile, gpu_controllers={
        **base.runtime_profile.gpu_controllers, "gpu1": base.runtime_profile.gpu_controllers["gpu0"]})
    return replace(base, hardware=hardware, runtime_profile=runtime)


def synthetic_cohort(case):
    builder = planner._TaskBuilder(RequestSpec("mixed", 0, 1, 1))
    start = builder.add("start", TaskCategory.POLICY)
    plan, router = planner._parallel_plan(case), planner._topology_router(case)
    prepared = planner._add_host_orchestration(builder, case, router, plan, (start,),
        name="mixed.prepare", request_count=3, token_count=3, include_gpu_transfer=False)
    groups = tuple(SimpleNamespace(group_id=g, request_ids=(g,), token_batch=1)
                   for g in ("cpu", "g0", "g01"))
    owners = {"cpu": ("cpu0",), "g0": ("gpu0",), "g01": ("gpu0", "gpu1")}
    tasks = {}
    for group in groups:
        prior = (prepared,)
        for owner in owners[group.group_id]:
            is_gpu = owner.startswith("gpu")
            task_id = builder.add(group.group_id + "." + owner, TaskCategory.COMPUTE,
                (ResourceDemand(owner + ".scalar", 10),), dependencies=prior, advance=False,
                metadata={"target_component": owner,
                    "layer_id": "layer-" + owner,
                    "operator_invocation_group_id": group.group_id,
                    "cost_model": {"model": "gpu_elementwise_roofline" if is_gpu else "cpu_elementwise_roofline"}})
            tasks[group.group_id, owner] = task_id
            prior = (task_id,)
    audit = planner._add_gpu_consumer_frontends(builder, case, router, plan, groups, (prepared,),
        name="mixed.frontend", execution_phase="prefill")
    lowering = planner._serving_lowering_from_builder(case, builder, cohort_id="mixed", kind="prefill",
        model="synthetic", assumptions=(), scenario_hash=None,
        extra_metadata={"token_batch": 3})
    summary = planner.execute_cost_schedule(lowering.schedule)
    facts = tuple({"group_id": group.group_id, "request_ids": group.request_ids}
                  for group in groups)
    stages, reason = planner._compact_execution_stages(case, summary.execution_records, facts)
    return lowering.schedule, summary, stages, reason, tasks, audit


def test_per_gpu_group_counts_and_independent_cpu_stage_do_not_wait_for_gpu():
    case = two_gpu_scenario()
    schedule, summary, stages, reason, tasks, audit = synthetic_cohort(case)
    assert reason is None and stages
    seen = set()
    for task in schedule.tasks:
        assert set(task.dependencies) <= seen
        seen.add(task.task_id)
    assert {k: v["submission_count"] for k, v in audit["by_gpu"].items()} == {"gpu0": 2, "gpu1": 1}
    assert audit["cpu_only_invocation_group_ids"] == ("cpu",)
    front_ids = {t.task_id for t in schedule.tasks if t.metadata.get("orchestration_stage") == FRONT}
    assert not front_ids.intersection(ancestors(schedule.tasks, tasks["cpu", "cpu0"]))
    front_stages = {s["stage_id"] for s in stages if s.get("stage_role") == FRONT}
    cpu_stage = next(s for s in stages if s.get("group_id") == "cpu")
    assert not front_stages.intersection(cpu_stage["dependencies"])
    slower = replace(case, runtime_profile=replace(case.runtime_profile, gpu_controllers={
        key: replace(value, command_processor=replace(value.command_processor,
                     command_submission_latency_ns=1e8))
        for key, value in case.runtime_profile.gpu_controllers.items()}))
    changed, changed_summary, changed_stages, changed_reason, changed_tasks, _ = synthetic_cohort(slower)
    assert changed_reason is None
    cpu_end = next(r.end_ns for r in summary.execution_records if r.task_id == tasks["cpu", "cpu0"])
    changed_end = next(r.end_ns for r in changed_summary.execution_records if r.task_id == changed_tasks["cpu", "cpu0"])
    assert changed_end == cpu_end
    assert next(s for s in changed_stages if s.get("group_id") == "cpu") == cpu_stage


def replay_in_real_online_executor(case, cohort):
    from heterollm_sim import serving
    plan = serving.compile_serving_plan(case)
    cohort = replace(cohort, items=tuple(replace(item, request_id=plan.requests[0].request_id)
                                        for item in cohort.items))
    lowering, detail, raw_stages, reason = lowered(case, cohort)
    assert reason is None
    production = planner.estimate_serving_cohort_cost(case, cohort)
    assert production["duration_ns"] == detail.makespan_ns
    assert production["metadata"]["execution_stages"] == raw_stages
    stages, parse_reason = serving._execution_stages_from_metadata(production["metadata"])
    assert parse_reason is None
    runtime = serving._OnlineRuntime(plan, lambda *_: serving.BatchCost(1.0))
    cost = serving.BatchCost(production["duration_ns"], metadata=production["metadata"])
    executed = runtime._schedule_execution_stages(
        cohort, cost, stages, 0.0, detail.makespan_ns, 0.0)
    return lowering, detail, raw_stages, executed


def test_first_qkv_h2d_waits_without_delaying_cpu_prefix_and_online_replays_exactly():
    case = offload_scenario(64)
    slow = replace(case, runtime_profile=replace(case.runtime_profile, gpu_controllers={
        key: replace(value, command_processor=replace(value.command_processor,
                     command_submission_latency_ns=1e6))
        for key, value in case.runtime_profile.gpu_controllers.items()}))
    original = replay_in_real_online_executor(case, _cohort(64, 1))
    delayed = replay_in_real_online_executor(slow, _cohort(64, 1))
    original_records = {r.task_id: r for r in original[1].execution_records}
    delayed_records = {r.task_id: r for r in delayed[1].execution_records}
    prefix = next(s for s in original[2] if s.get("stage_role") == "frontend_cpu_prefix")
    assert prefix["frontend_wait_refinement"]["prefix_stage_count"] == 1
    for task in prefix["execution_tasks"]:
        before, after = original_records[task["task_id"]], delayed_records[task["task_id"]]
        assert (before.start_ns, before.end_ns) == (after.start_ns, after.end_ns)
    for lowering, detail, stages, online in (original, delayed):
        ready = next(r for r in detail.execution_records
                     if r.metadata.get("event_kind") == "gpu_command_processor_batch")
        inputs = [r for r in detail.execution_records
                  if r.metadata.get("event_kind") == "operator_input_transfer"
                  and r.metadata.get("source_component") == "cpu0"
                  and r.metadata.get("target_component") == "gpu0"
                  and r.metadata.get("projection_id") in {"attention.q", "attention.k", "attention.v"}]
        assert len(inputs) == 3
        assert all(sum(d.bytes_moved for d in r.demands) == 32768 for r in inputs)
        assert all(r.start_ns >= ready.end_ns for r in inputs)
        assert all(ready.task_id in ancestors(lowering.schedule.tasks, r.task_id) for r in inputs)
        assert online[3] == pytest.approx(detail.makespan_ns, rel=0, abs=1e-7)
        assert online[-1] == pytest.approx(detail.makespan_ns, rel=0, abs=1e-7)
        assert len(stages) < 12
    before_prefix = next(row for row in original[3][6] if row["stage_id"] == prefix["stage_id"])
    after_prefix = next(row for row in delayed[3][6] if row["stage_id"] == prefix["stage_id"])
    assert (before_prefix["start_ns"], before_prefix["end_ns"]) == (after_prefix["start_ns"], after_prefix["end_ns"])


def test_overlapping_startup_prefix_uses_bounded_detail_and_exact_online_replay(monkeypatch):
    def forked_body(builder, scenario, plan, router, group, *, phase, dependencies):
        group_metadata = planner._serving_invocation_group_task_metadata(group)
        def add(name, component, duration, deps, *, resource=None, category=TaskCategory.COMPUTE, extra=None):
            return builder.add(phase + "." + name, category,
                (ResourceDemand(resource or component + ".scalar", duration),) if duration else (),
                dependencies=deps, advance=False, metadata={
                    **group_metadata, "target_component": component,
                    "layer_id": "gpu_body" if component == "gpu0" else ("cpu_tail" if name == "tail" else "cpu_prefix"),
                    **({"cost_model": {"model": "gpu_elementwise_roofline" if component == "gpu0" else "cpu_elementwise_roofline"}}
                       if category == TaskCategory.COMPUTE else {}),
                    **(extra or {}),
                })
        a = add("a", "cpu0", 1000, dependencies, resource="cpu0.pipeline")
        b = add("b", "cpu0", 20000, dependencies, resource="cpu0.vector")
        zero = add("a_ready", "cpu0", 0, (a,), category=TaskCategory.SYNCHRONIZATION)
        upload = add("first_input", "gpu0", 1000, (zero,), category=TaskCategory.COMMUNICATION,
                     resource="link.cpu-gpu-pcie.cpu0->gpu0", extra={
                         "event_kind": "operator_input_transfer", "source_component": "cpu0", "transfer_kind": "data"})
        gpu = add("gpu", "gpu0", 5000, (upload,))
        joined = add("join", "cpu0", 0, (b, gpu), category=TaskCategory.SYNCHRONIZATION)
        return add("tail", "cpu0", 100, (joined,), resource="cpu0.pipeline")
    monkeypatch.setattr(planner, "_compile_or_replay_serving_invocation", forked_body)
    lowering, detail, stages, online = replay_in_real_online_executor(offload_scenario(64), _cohort(64, 1))
    refined = [s for s in stages if s.get("frontend_wait_refinement")]
    assert refined
    assert all(s["frontend_wait_refinement"]["representation"] == "bounded_detailed_initial_region" for s in refined)
    assert len(refined) <= 5 and len(stages) < 12
    assert any(s["frontend_wait_refinement"]["zero_service_dependency_markers"] for s in refined)
    assert online[3] == pytest.approx(detail.makespan_ns, rel=0, abs=1e-7)
    assert online[-1] == pytest.approx(detail.makespan_ns, rel=0, abs=1e-7)
    cpu_b = next(r for r in detail.execution_records if r.task_id.endswith(".b"))
    gpu = next(r for r in detail.execution_records if r.task_id.endswith(".gpu"))
    assert cpu_b.start_ns < gpu.start_ns < gpu.end_ns < cpu_b.end_ns
    next_batch = replay_in_real_online_executor(offload_scenario(64),
        _cohort(64, 1, name="different-physical-batch", context=64))
    def indices_by_source(rows):
        return {str(row["execution_tasks"][0]["task_id"]).rsplit(".", 1)[-1]: row["stage_index"]
                for row in rows if row.get("stage_role") == "frontend_wait_region"}
    assert indices_by_source(stages) == indices_by_source(next_batch[2])


def test_transfer_gates_require_qualified_group_and_device_and_exclude_cohort_payload():
    case = two_gpu_scenario()
    builder = planner._TaskBuilder(RequestSpec("transfer-scope", 0, 1, 1))
    plan, router = planner._parallel_plan(case), planner._topology_router(case)
    prepared = planner._add_host_orchestration(builder, case, router, plan, (),
        name="prepare", request_count=2, token_count=2, include_gpu_transfer=False)
    builder.add("gpu_body", TaskCategory.COMPUTE, (ResourceDemand("gpu0.scalar", 1),),
        dependencies=(prepared,), advance=False, metadata={"operator_invocation_group_id": "gpu-group",
            "target_component": "gpu0", "cost_model": {"model": "gpu_elementwise_roofline"}})
    transfers = {}
    for label, group, target in (("qualified", "gpu-group", "gpu0"),
                                  ("other-device", "gpu-group", "gpu1"),
                                  ("cpu-group", "cpu-group", "gpu0")):
        transfers[label] = builder.add(label, TaskCategory.COMMUNICATION,
            (ResourceDemand("link.cpu-gpu-pcie.cpu0->" + target, 10, bytes_moved=16),),
            dependencies=(prepared,), advance=False, metadata={
                "operator_invocation_group_id": group, "target_component": target,
                "source_component": "cpu0", "transfer_kind": "data", "event_kind": "operator_input_transfer"})
    groups = tuple(SimpleNamespace(group_id=group, request_ids=(group,), token_batch=1)
                   for group in ("gpu-group", "cpu-group"))
    audit = planner._add_gpu_consumer_frontends(builder, case, router, plan, groups, (prepared,),
        name="frontend", execution_phase="prefill")
    assert set(audit["by_gpu"]) == {"gpu0"}
    ready = next(t.task_id for t in builder.tasks if t.metadata.get("event_kind") == "gpu_command_processor_batch")
    assert ready in ancestors(builder.tasks, transfers["qualified"])
    assert ready not in ancestors(builder.tasks, transfers["other-device"])
    assert ready not in ancestors(builder.tasks, transfers["cpu-group"])
    assert ready not in ancestors(builder.tasks, prepared)
    for task in builder.tasks:
        if task.metadata.get("event_kind") == "host_cohort_h2d":
            assert ready not in ancestors(builder.tasks, task.task_id)


def test_two_prefill_batches_keep_prefix_readiness_separate_from_model_h2d():
    from heterollm_sim import serving
    case = offload_scenario(64)
    case = replace(case, workload=replace(case.workload,
        requests=(RequestSpec("request-0000", 0, 128, 1),)))
    plan = serving.compile_serving_plan(case)
    runtime = serving._OnlineRuntime(plan, lambda *_: serving.BatchCost(1.0))
    observed_ready = []
    original_cached = runtime._replay_cached_execution_stage_tasks
    original_uncached = runtime._replay_execution_stage_tasks
    def cached(*args, **kwargs):
        observed_ready.append(dict(args[1]))
        return original_cached(*args, **kwargs)
    def uncached(*args, **kwargs):
        observed_ready.append(dict(args[2]))
        return original_uncached(*args, **kwargs)
    results = []
    metadata_rows = []
    with patch.object(runtime, "_replay_cached_execution_stage_tasks", side_effect=cached), \
         patch.object(runtime, "_replay_execution_stage_tasks", side_effect=uncached):
        for index in range(2):
            cohort = _cohort(64, int(index == 1), name="physical-prefill-" + str(index), context=64 * index)
            cohort = replace(cohort, items=tuple(replace(item, request_id="request-0000") for item in cohort.items))
            production = planner.estimate_serving_cohort_cost(case, cohort)
            stages, reason = serving._execution_stages_from_metadata(production["metadata"])
            assert reason is None
            result = runtime._schedule_execution_stages(cohort,
                serving.BatchCost(production["duration_ns"], metadata=production["metadata"]),
                stages, 0.0, production["duration_ns"], 0.0)
            results.append(result)
            rows = production["metadata"]["execution_stages"]
            metadata_rows.append(rows)
            prefix = next(row for row in result[6] if ".frontend_prefix." in row["stage_id"])
            model = next(row for row in result[6] if row["stage_id"].endswith(".stage0000")
                         and row["invocation_group_id"] is not None)
            assert model["stage_index"] == 0
            assert prefix["stage_index"] != model["stage_index"]
            prefix_key = ("request-0000", prefix["component_id"], prefix["stage_index"])
            assert runtime._request_stage_ready_ns[prefix_key] == prefix["end_ns"]
            assert runtime._request_stage_ready_ns[("request-0000", model["component_id"], 0)] == model["end_ns"]
            assert prefix["end_ns"] < model["start_ns"]
    first_prefix = next(row for row in results[0][6] if ".frontend_prefix." in row["stage_id"])
    second_prefix = next(row for row in results[1][6] if ".frontend_prefix." in row["stage_id"])
    first_model = next(row for row in results[0][6] if row["stage_id"].endswith(".stage0000")
                       and row["invocation_group_id"] is not None)
    assert len(metadata_rows[0]) != len(metadata_rows[1])  # output selection changes the tail
    assert first_prefix["stage_index"] == second_prefix["stage_index"]
    assert observed_ready[1][second_prefix["stage_id"]] == first_prefix["end_ns"]
    assert observed_ready[1][second_prefix["stage_id"]] < first_model["end_ns"]
    first_front = next(row for row in results[0][6] if row["stage_id"].startswith("serving.gpu_consumer_frontend."))
    second_front = next(row for row in results[1][6] if row["stage_id"].startswith("serving.gpu_consumer_frontend."))
    assert first_front["stage_index"] == second_front["stage_index"]
    assert observed_ready[1][second_front["stage_id"]] == first_front["end_ns"]
    assert not any(row["stage_index"] == first_front["stage_index"] and row["component_id"] == first_front["component_id"]
                   for row in metadata_rows[0] if row.get("group_id") is not None)
