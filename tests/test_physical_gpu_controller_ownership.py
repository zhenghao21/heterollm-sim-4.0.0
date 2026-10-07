"""Physical DRAM owns controller latency; zero traffic cannot select unrelated bytes."""
from types import SimpleNamespace

import pytest

from heterollm_sim import planner as p
from heterollm_sim.contracts import ResourceDemand, _PreparedExecutionStage as Stage, _PreparedExecutionTask as Task
from heterollm_sim.serving import _OnlineRuntime, BatchCohort, BatchItem
from test_direct_backing_physical_ownership import prepared_case


def overlay(scenario, demands, metadata=None):
    task = Task("gpu-work", (), ("request-0000",), demands, metadata=metadata or {})
    stage = Stage("root", 0, (), ("request-0000",), "gpu0", 10.0, (task,))
    runtime = SimpleNamespace(plan=SimpleNamespace(scenario=scenario), _gpu_controller_resource_domains={})
    cohort = BatchCohort("cohort-0", "decode", 0.0, (BatchItem("request-0000", "decode", 1, 16),))
    result = _OnlineRuntime._with_gpu_controller_stages(runtime, (stage,), cohort)
    return runtime, result


@pytest.mark.parametrize("hardware", ("local", "b200_2hbf", "b200_3hbf"))
def test_physical_presets_have_no_second_vram_latency_or_observer_traffic(hardware):
    scenario = prepared_case(hardware)
    rank = p._parallel_plan(scenario).ranks[0]
    assert rank.memory_component_id is None
    resource = p._rank_memory_resource(scenario, rank)
    runtime, stages = overlay(scenario, (ResourceDemand(resource, 100, bytes_moved=4096),
        ResourceDemand("gpu0.l2", 5, bytes_moved=8192)),
        {"physical_memory_config": {"kind": "DRAM"}, "memory_accesses":
            ({"operation": "read", "byte_count": 4096},)})
    assert resource in runtime._gpu_controller_resource_domains["gpu0"][1]
    tasks = [task for stage in stages for task in stage.execution_tasks]
    assert any(task.metadata.get("service_domain") == "gpu_address_translation" for task in tasks)
    assert not any(d.resource_id.endswith((".vram_controller", ".l2_controller")) for task in tasks for d in task.demands)
    root = next(stage for stage in stages if stage.stage_id == "root")
    assert root.service_ns == 10.0


def test_unrelated_compute_bytes_do_not_create_memory_or_translation_work():
    scenario = prepared_case()
    _runtime, stages = overlay(scenario, (ResourceDemand("gpu0.tensor", 10, bytes_moved=1000000),))
    assert len(stages) == 1
    assert len(stages[0].execution_tasks) == 1


def test_zero_declared_memory_is_zero_not_a_max_resource_fallback():
    scenario = prepared_case()
    resource = p._rank_memory_resource(scenario, p._parallel_plan(scenario).ranks[0])
    _runtime, stages = overlay(scenario, (ResourceDemand(resource, 0),
        ResourceDemand("gpu0.tensor", 10, bytes_moved=1000000)),
        {"physical_memory_config": {"kind": "DRAM"}, "memory_accesses": ()})
    assert len(stages) == 1 and len(stages[0].execution_tasks) == 1


def test_nonphysical_custom_memory_domain_remains_explicit():
    scenario = prepared_case()
    resource = p._rank_memory_resource(scenario, p._parallel_plan(scenario).ranks[0])
    _runtime, stages = overlay(scenario, (ResourceDemand(resource, 10, bytes_moved=4096),))
    memory = [task for stage in stages for task in stage.execution_tasks
              if task.metadata.get("service_role") == "explicit_nonphysical_controller_owner"]
    assert len(memory) == 1
    assert memory[0].demands[0].bytes_moved == 4096
    assert memory[0].demands[0].service_ns == 300
