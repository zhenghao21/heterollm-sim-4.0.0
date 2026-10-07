from dataclasses import replace

import pytest

from heterollm_sim.communication import TopologyRouter
from heterollm_sim.ir import RequestSpec
from heterollm_sim.parallel import build_parallel_plan
from heterollm_sim import planner
from heterollm_sim.reference import build_reference_scenario


@pytest.mark.parametrize(
    "memory,target,accounting",
    (
        (None, "hbm1", "explicit_remote_transfer"),
        ("hbm1", "hbm1", "included_in_attention_kernel"),
        (None, "cim0", "included_in_attention_kernel"),
    ),
)
def test_cim_kv_access_preserves_remote_explicit_and_same_endpoint_pairs(
    monkeypatch, memory, target, accounting,
):
    scenario = build_reference_scenario()
    plan = build_parallel_plan(scenario)
    rank = replace(plan.ranks[0], component_id="cim0", memory_component_id=memory)
    builder = planner._TaskBuilder(RequestSpec("cim-kv", 0.0, 1, 1))

    def gpu_cpu_only(*args, **kwargs):
        pytest.fail("CIM KV access must not infer a CPU/GPU runtime backend")

    monkeypatch.setattr(planner, "_compute_local_runtime_memory_component_id", gpu_cpu_only)
    with planner._compilation_scope(scenario):
        task_id = planner._add_kv_access(
            builder, scenario, planner._topology_router(scenario), plan, rank,
            "cim0", target, 64, (), name="kv.append", metadata={"memory_direction": "write"},
        )
    assert task_id == builder.tasks[-1].task_id
    assert builder.tasks[-1].metadata["resource_accounting"] == accounting
    if accounting == "explicit_remote_transfer":
        assert any(task.demands for task in builder.tasks)
    else:
        assert len(builder.tasks) == 1
        assert not builder.tasks[0].demands
