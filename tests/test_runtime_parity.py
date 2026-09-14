from heterollm_sim.contracts import RunManifest
from heterollm_sim.runtime_adapters import LlamaCppAdapter, LlamaCppRuntimeConfig
from heterollm_sim.engine import ScheduleIR
from heterollm_sim.contracts import TaskSpec, TaskCategory, ResourceDemand
from heterollm_sim.engine import simulate_schedule
import pytest


def _manifest():
    return RunManifest(
        schema_version="4.0.0", run_id="r", random_seed=1,
        simulator_version="test", model_name="m", hardware_name="h",
        workload_name="w",
    )


def test_llama_config_fingerprint_changes_with_every_runtime_knob():
    base = LlamaCppRuntimeConfig()
    changed = LlamaCppRuntimeConfig(**{**base.to_dict(), "threads_batch": base.threads_batch + 1})
    assert base.fingerprint != changed.fingerprint
    changed = LlamaCppRuntimeConfig(**{**base.to_dict(), "numa": "isolate", "cpu_range": "0-15"})
    assert base.fingerprint != changed.fingerprint


def test_llama_adapter_stamps_typed_semantics_and_manifest_fingerprint():
    config = LlamaCppRuntimeConfig(
        threads=16, threads_batch=12, batch=64, ubatch=32, context=512,
        parallel=1, gpu_layers=-1, flash_attn=True, kv_type_k="q8_0",
        kv_type_v="q8_0", kv_unified=True, cont_batching=True,
        warmup=False, seed=42,
    )
    plan = LlamaCppAdapter().lower((), config=config)
    assert plan.semantics["threads_batch"] == 12
    assert plan.semantics["flash_attn"] is True
    assert plan.semantics["config_fingerprint"] == config.fingerprint
    schedule = plan.to_schedule(_manifest())
    assert schedule.manifest.metadata["runtime_fingerprint"] == plan.fingerprint
    assert schedule.manifest.metadata["runtime_semantics"]["seed"] == 42


def test_legacy_llama_adapter_arguments_remain_supported():
    plan = LlamaCppAdapter().lower(
        (), batch_size=64, ubatch_size=32, parallel=1,
        gpu_layers=-1, context_length=128,
    )
    assert plan.semantics["logical_batch_size"] == 64
    assert plan.semantics["physical_ubatch_size"] == 32

def test_runtime_adapter_schedule_preserves_explicit_resource_capacity():
    tasks=tuple(TaskSpec(task_id=str(i),request_id='r',name=str(i),category=TaskCategory.COMPUTE,demands=(ResourceDemand('gpu',10),)) for i in range(2))
    schedule=LlamaCppAdapter().lower(tasks).to_schedule(_manifest(),resource_capacities={'gpu':2})
    assert schedule.resource_capacities == {'gpu':2}
    assert simulate_schedule(schedule).makespan_ns == 10

def test_config_and_explicit_op_offload_conflict_is_rejected():
    with pytest.raises(ValueError):
        LlamaCppAdapter().lower((),config=LlamaCppRuntimeConfig(),op_offload=False)
