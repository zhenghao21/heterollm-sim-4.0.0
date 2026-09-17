"""Controller access waves follow GPU memory consumers, never command staging."""
from types import SimpleNamespace
from heterollm_sim.contracts import (ResourceDemand, _PreparedExecutionStage as Stage,
                                   _PreparedExecutionTask as Task)
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serving import _OnlineRuntime, BatchCohort, BatchItem


def fixture(frontend=True, mixed=False):
    runtime = SimpleNamespace(plan=SimpleNamespace(scenario=build_reference_scenario()),
        _gpu_controller_resource_domains={"gpu0": ("gpu0.l2", frozenset({"hbm0.hbm_fabric"}))})
    host = Task("host_build", (), ("r",),
        (ResourceDemand("cpu0.pipeline", 1000, work_units=128),),
        metadata={"orchestration_stage": "gpu_consumer_frontend", "event_kind": "cpu_invocation_command_build"})
    payload = Task("h2d", ("host_build",), ("r",),
        (ResourceDemand("link.cpu-gpu-pcie.cpu0->gpu0", 50, bytes_moved=8),),
        metadata={"orchestration_stage": "gpu_consumer_frontend", "event_kind": "host_cohort_h2d"})
    memory = Task("matmul", (), ("r",),
        (ResourceDemand("hbm0.hbm_fabric", 10000, bytes_moved=1 << 25),
         ResourceDemand("gpu0.l2", 1000, bytes_moved=1 << 20),
         ResourceDemand("gpu0.tensor_core", 5000, work_units=10000)),
        metadata={"cost_model": {"model": "gpu_hbm_roofline"}, "event_kind": "mlp_projection"})
    prefix = Stage("frontend", -1000011, (), ("r",), "gpu0", 1050, (host, payload))
    operator = Stage("operator", 0, ("frontend",) if frontend else (), ("r",), "gpu0", 10000,
        (host, payload, memory) if mixed else (memory,))
    stages = (prefix, operator) if frontend else (operator,)
    cohort = BatchCohort("proof", "decode", 0, (BatchItem("r", "decode", 1, 128),))
    return runtime, stages, cohort


def overlay(frontend=True, mixed=False):
    runtime, stages, cohort = fixture(frontend, mixed)
    return stages, _OnlineRuntime._with_gpu_controller_stages(runtime, stages, cohort)


def memory_stages(stages):
    return [stage for stage in stages if any(task.metadata.get("event_kind") == "gpu_memory_controller_pipeline"
                                             for task in stage.execution_tasks)]


def test_orchestration_only_prefix_does_not_own_operator_memory_waves():
    original, lowered = overlay()
    by_id = {stage.stage_id: stage for stage in lowered}
    assert by_id["frontend"] == original[0]
    assert [stage.stage_id for stage in memory_stages(lowered)] == ["operator"]
    # The MMU prefix is causal to actual data execution and retains its input
    # dependency. The front-end must not wait for all future HBM access waves.
    mmu = [stage for stage in lowered if any(task.metadata.get("event_kind") == "gpu_mmu_tlb_batch"
                                          for task in stage.execution_tasks)]
    assert len(mmu) == 1 and mmu[0].dependencies == ("frontend",)
    assert by_id["operator"].dependencies == (mmu[0].stage_id,)


def test_control_prefix_does_not_change_memory_quantity_or_latency():
    _, with_prefix = overlay()
    _, without_prefix = overlay(False)
    def controller_demands(stages):
        return [demand for stage in stages for task in stage.execution_tasks
                if task.metadata.get("event_kind") == "gpu_memory_controller_pipeline"
                for demand in task.demands]
    assert controller_demands(with_prefix) == controller_demands(without_prefix)
    assert memory_stages(with_prefix)[0].service_ns == memory_stages(without_prefix)[0].service_ns


def test_mixed_control_and_real_memory_stage_still_gets_overlay():
    _, lowered = overlay(False, mixed=True)
    assert [stage.stage_id for stage in memory_stages(lowered)] == ["operator"]


def test_frontend_without_device_body_does_not_invent_memory_access():
    runtime, stages, cohort = fixture()
    assert _OnlineRuntime._with_gpu_controller_stages(runtime, (stages[0],), cohort) == (stages[0],)


def test_real_lowered_resource_ownership_survives_stage_handoff():
    from heterollm_sim.serving import _execution_stages_from_metadata
    from heterollm_sim import planner
    from tests.test_final_layer_output_selection_planner import _scenario, _cohort
    case = _scenario("qwen2_decoder")
    cohort = _cohort(1, 1, phase="decode", context=128)
    # Actual-consumer binding deliberately retains original task creation
    # order; the event kernel validates the dependency DAG, including forward
    # references. Do not impose an unrelated list-order assumption here.
    with planner._compilation_scope(case):
        lowered = planner._lower_serving_cohort(case, cohort)
    summary = planner.execute_cost_schedule(lowered.schedule)
    raw_stages, reason = planner._compact_execution_stages(
        case, summary.execution_records, lowered.extra_metadata["operator_invocation_groups"])
    assert reason is None
    stages, reason = _execution_stages_from_metadata({
        "execution_stages": raw_stages,
        "execution_stage_source": "executed_task_dag_kernel_timeline"})
    assert reason is None
    front = next(stage for stage in stages if "gpu_consumer_frontend" in stage.stage_id)
    # The compact wire form intentionally omits some task metadata. The
    # classifier must use actual resource ownership, not labels lost here.
    assert front.execution_tasks
    assert any(not task.metadata for task in front.execution_tasks)
    runtime = SimpleNamespace(plan=SimpleNamespace(scenario=case), _gpu_controller_resource_domains={})
    result = _OnlineRuntime._with_gpu_controller_stages(runtime, stages, cohort)
    assert next(stage for stage in result if stage.stage_id == front.stage_id) == front
    owned = memory_stages(result)
    assert owned and all(stage.stage_id != front.stage_id for stage in owned)


def test_nonstandard_backing_resource_is_recognized_without_gpu_prefix():
    # Physical profile ownership, not spelling, qualifies a memory-only task.
    runtime, _, cohort = fixture(False)
    runtime._gpu_controller_resource_domains = {"gpu0": ("shared.cache", frozenset({"external.memory"}))}
    task = Task("copy", (), ("r",), (ResourceDemand("external.memory", 10000, bytes_moved=1 << 20),))
    stage = Stage("data_only", 0, (), ("r",), "gpu0", 10000, (task,))
    result = _OnlineRuntime._with_gpu_controller_stages(runtime, (stage,), cohort)
    assert [item.stage_id for item in memory_stages(result)] == ["data_only"]
    vram = next(demand for item in result for work in item.execution_tasks
                if work.metadata.get("event_kind") == "gpu_memory_controller_pipeline"
                for demand in work.demands if demand.resource_id == "gpu0.vram_controller")
    assert vram.bytes_moved == 1 << 20


def test_cpu_work_does_not_borrow_zero_work_gpu_demand_to_qualify():
    runtime, _, cohort = fixture(False)
    task = Task("control", (), ("r",), (
        ResourceDemand("cpu0.pipeline", 1000, work_units=128),
        ResourceDemand("gpu0.scalar", 0),))
    stage = Stage("control_only", 0, (), ("r",), "gpu0", 1000, (task,))
    assert _OnlineRuntime._with_gpu_controller_stages(runtime, (stage,), cohort) == (stage,)


def test_online_executor_overlaps_controller_waves_with_actual_operator():
    from dataclasses import replace
    from heterollm_sim import serving
    from heterollm_sim.ir import RequestSpec
    fake, stages, cohort = fixture()
    case = fake.plan.scenario
    case = replace(case, workload=replace(case.workload,
        requests=(RequestSpec("r", 0, 1, 1),), request_count=1,
        prompt_tokens=1, output_tokens=1))
    runtime = serving._OnlineRuntime(serving.compile_serving_plan(case),
                                    lambda *_: serving.BatchCost(1.0))
    runtime._gpu_controller_resource_domains = fake._gpu_controller_resource_domains
    overlaid = runtime._with_gpu_controller_stages(stages, cohort)
    metadata = {"execution_stage_makespan_ns": 11050.0,
                "execution_stage_source": "executed_task_dag_kernel_timeline",
                "execution_stages_include_host_orchestration": True}
    result = runtime._schedule_execution_stages(cohort,
        serving.BatchCost(11050.0, metadata=metadata), overlaid,
        0.0, 11050.0, 0.0, runtime_controller_stage_count=1)
    # 1050 command/payload -> 300 translation -> max(10000 operator,
    # 19200 controller waves). No LLM timing is involved in this tiny graph.
    assert result[-1] == 1050.0 + 300.0 + max(10000.0, 19200.0)
    times = {item["stage_id"]: item for item in result[6]}
    assert times["frontend"]["end_ns"] == 1050.0
    assert times["operator"]["start_ns"] == 1350.0
