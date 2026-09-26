from dataclasses import replace

import pytest

from heterollm_sim.ir import SchedulerSpec
from heterollm_sim.llama_scenario import apply_llama_runtime_config
from heterollm_sim.reporting import run_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
from heterollm_sim.llama_trace_diff import diff_scheduler_traces
from tests.test_llama_mixed_batching import authored


def test_llama_profile_resolves_all_scheduler_fields_and_is_idempotent():
    scenario = authored()
    authored_scheduler = replace(
        scenario.workload.scheduler,
        policy="decode_first_aging",
        preemption_enabled=True,
        prefill_chunk_tokens=8,
    )
    scenario = replace(
        scenario,
        workload=replace(scenario.workload, scheduler=authored_scheduler),
    )
    config = LlamaCppRuntimeConfig(batch=64, ubatch=16, context=512, parallel=2)
    lowered = apply_llama_runtime_config(scenario, config, materialize_placement=False)
    assert lowered.workload.scheduler.policy == "decode_first"
    assert lowered.workload.scheduler.preemption_enabled is False
    assert lowered.workload.scheduler.prefill_chunk_tokens == 64
    assert lowered.workload.metadata["llama_cpp_effective_scheduler_policy"]["physical_ubatch_tokens"] == 16
    repeated = apply_llama_runtime_config(lowered, config, materialize_placement=False)
    assert repeated.workload.scheduler == lowered.workload.scheduler
    assert repeated.workload.metadata["llama_cpp_effective_scheduler_policy"] == lowered.workload.metadata[
        "llama_cpp_effective_scheduler_policy"
    ]


def test_generic_profile_preserves_authored_scheduler_policy():
    scenario = authored()
    scheduler = replace(
        scenario.workload.scheduler,
        policy="decode_first_aging",
        preemption_enabled=True,
        prefill_chunk_tokens=8,
    )
    scenario = replace(scenario, workload=replace(scenario.workload, scheduler=scheduler))
    lowered = apply_llama_runtime_config(
        scenario,
        LlamaCppRuntimeConfig(
            batch=64,
            ubatch=16,
            context=512,
            parallel=2,
            policy="generic",
        ),
        materialize_placement=False,
    )
    assert lowered.workload.scheduler.policy == "decode_first_aging"
    assert lowered.workload.scheduler.preemption_enabled is True
    assert lowered.workload.scheduler.prefill_chunk_tokens == 8


def test_llama_profile_rejects_explicit_preemption_override():
    with pytest.raises(ValueError, match="preemption_enabled=True"):
        apply_llama_runtime_config(
            authored(),
            LlamaCppRuntimeConfig(
                batch=64,
                ubatch=16,
                context=512,
                parallel=2,
                preemption_enabled=True,
            ),
            materialize_placement=False,
        )


def test_serving_trace_binds_slot_decision_to_physical_ubatches():
    scenario = apply_llama_runtime_config(
        authored(),
        LlamaCppRuntimeConfig(batch=64, ubatch=16, context=512, parallel=2),
        materialize_placement=False,
    )
    result = run_scenario(scenario)
    decisions = [row for row in result.serving.schedule_trace if row["kind"] == "decision"]
    assert decisions and decisions[0]["logical_batch"]["rows"][0]["slot_id"] == 0
    plan = result.serving.batches[0].metadata["execution_plan"]
    assert plan["schema"] == "heterollm.execution-plan/v1"
    assert len(plan["physical_ubatches"]) == 4
    assert sum(group["rows"] for group in plan["physical_ubatches"]) == 64
    assert sum(group["append_tokens"] for group in plan["kv_ranges"]) == 64
    assert any(row["kind"] == "state_commit" for row in result.serving.schedule_trace)


def test_runtime_identity_is_bound_without_claiming_native_timing():
    identity = {
        "version": "b10760",
        "source_sha256": {"server-context.cpp": "a" * 64},
        "binary_sha256": {"llama-server.exe": "b" * 64},
        "build_options": {"cuda": True},
        "native_trace_validated": True,
    }
    scenario = apply_llama_runtime_config(
        authored(),
        LlamaCppRuntimeConfig(batch=64, ubatch=64, context=512, parallel=2),
        materialize_placement=False,
        runtime_identity=identity,
    )
    evidence = scenario.workload.metadata["llama_cpp_runtime_identity"]
    assert evidence["status"] == "bound"
    assert evidence["native_trace_validated"] is True
    assert evidence["timing_validated"] is False


def test_execution_failure_restores_request_and_capacity_state():
    scenario = apply_llama_runtime_config(
        authored(),
        LlamaCppRuntimeConfig(batch=64, ubatch=64, context=512, parallel=2),
        materialize_placement=False,
    )

    def fail(_scenario, _cohort):
        raise RuntimeError("injected lowerer failure")

    with pytest.raises(RuntimeError, match="injected lowerer failure"):
        run_scenario(scenario, batch_lowerer=fail)


def test_scheduler_trace_diff_reports_first_ubatch_divergence():
    left = ({
        "kind": "execution_complete",
        "execution_plan": {
            "physical_ubatches": ({"rows": 8},),
            "kv_ranges": (),
            "execution_dependencies": (),
        },
    },)
    right = ({
        "kind": "execution_complete",
        "execution_plan": {
            "physical_ubatches": ({"rows": 4},),
            "kv_ranges": (),
            "execution_dependencies": (),
        },
    },)
    diff = diff_scheduler_traces(left, right)
    assert diff["equal"] is False
    assert diff["first_divergence"] == 0
    assert diff["category"] == "ubatch"


def test_scheduler_trace_diff_keeps_native_time_out_of_structure_comparison():
    left = ({"kind": "decision", "time_ns": 1.0, "logical_batch": {"rows": ()}},)
    right = ({"kind": "decision", "time_ns": 2.0, "logical_batch": {"rows": ()}},)
    diff = diff_scheduler_traces(left, right)
    assert diff["category"] == "timing"
