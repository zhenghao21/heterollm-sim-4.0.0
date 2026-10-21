"""One-sided composition needs a state-independent native issue order."""
from dataclasses import replace
from types import SimpleNamespace

from heterollm_sim.contracts import (_PreparedExecutionStage, _PreparedExecutionTask,
    ResourceDemand, TaskCategory)
from heterollm_sim.native_issue_order import native_issue_order_proof
from test_native_collective_row_reuse import fixture


def _stage(identity, tasks, dependencies=()):
    return _PreparedExecutionStage(identity, 0, tuple(dependencies), ("cohort-r",), "gpu", 0., tuple(tasks))


def _rows(stages):
    rows = []
    for stage in stages:
        for task in stage.execution_tasks:
            md = task.metadata
            resources = {"@physical-owner:"+access.get("physical_owner", md.get("physical_owner"))
                         for access in md.get("memory_accesses", ())}
            resources |= {demand.resource_id for demand in task.demands if demand.service_ns > 0.}
            rows.append((stage.stage_id, task, resources))
    return rows


def _kernel(capacities=None):
    return SimpleNamespace(_execution_resource_kernel=SimpleNamespace(
        _physical_owner=lambda resource: resource, resource_capacities=dict(capacities or {})))


def _compute(identity, deps, resource, service=500.):
    return _PreparedExecutionTask(identity, tuple(deps), ("cohort-r",),
        (ResourceDemand(resource, service),), metadata={"layer_id": "layer-000"})


def _memory(identity, deps, owner, config, demands=(), operation="read"):
    return _PreparedExecutionTask(identity, tuple(deps), ("cohort-r",), tuple(demands),
        category=TaskCategory.MEMORY, metadata={"layer_id": "layer-000", "physical_owner": owner,
            "physical_memory_config": config, "memory_accesses": ({"physical_owner": owner,
                "resource_id": owner, "operation": operation, "byte_count": 256},)})


def _proof(stages, physical, native):
    return native_issue_order_proof(stages, _rows(stages), physical, native)


def test_closed_ring_reads_always_precede_same_round_writes():
    (runtime, _config, physical, _core), _ranks, tasks = fixture(ranks=2)
    proof = _proof((_stage("stage", tasks),), physical, runtime)
    assert proof["qualified"], proof
    assert proof["unordered_pairs_by_rule"]["dominated_release"] > 0
    assert not proof["unproven_pairs"]


def _spill(config, frontend="gpu.frontend"):
    gemm = _compute("qkv.gpu_gemm", (), "gpu.tensor_core")
    launches = [_compute(f"kv.{kind}.kernel_launch", (gemm.task_id,), frontend) for kind in "kv"]
    spills = [_memory(f"kv.{kind}.gpu_memory", (launch.task_id,), "hbm", config,
                      (ResourceDemand("gpu.l2", 4.),)) for kind, launch in zip("kv", launches)]
    return (gemm, *launches, *spills)


def test_kv_spill_order_is_fixed_by_its_serialized_single_lane_launches():
    (_runtime, config, physical, _core), _ranks, _tasks = fixture(ranks=2)
    proof = _proof((_stage("stage", _spill(config)),), physical, _kernel())
    assert proof["qualified"], proof
    assert proof["unordered_pairs_by_rule"] == {"same_release": 1, "serialized_release": 1}


def test_multi_lane_launches_do_not_order_their_spills():
    (_runtime, config, physical, _core), _ranks, _tasks = fixture(ranks=2)
    proof = _proof((_stage("stage", _spill(config)),), physical, _kernel({"gpu.frontend": 2}))
    assert not proof["qualified"]
    assert proof["reason"] == "native_issue_order_may_depend_on_initial_state"
    assert {row["tasks"][0].split(".")[-1] for row in proof["unproven_pairs"]} == {"gpu_memory"}


def test_parallel_pipelines_behind_memory_writes_may_swap_a_shared_host_queue():
    (_runtime, config, physical, _core), _ranks, _tasks = fixture(ranks=2)
    tasks = []
    for index in range(2):
        write = _memory(f"payload{index}.write", (), "hbm" if index == 0 else "hbm1", config,
                        operation="write")
        tasks += [write, _compute(f"payload{index}.command_build", (write.task_id,), "cpu.scheduler", 11.7)]
    proof = _proof((_stage("stage", tasks),), physical, _kernel())
    assert not proof["qualified"]
    assert proof["unproven_pairs"][0]["resource"] == "cpu.scheduler"


def test_same_release_needs_one_resource_group():
    (_runtime, config, physical, _core), _ranks, _tasks = fixture(ranks=2)
    root = _compute("root", (), "gpu.tensor_core")
    left = _compute("left", (root.task_id,), "cpu.scheduler")
    right = _PreparedExecutionTask("right", (root.task_id,), ("cohort-r",),
        (ResourceDemand("cpu.scheduler", 500.), ResourceDemand("cpu.dma", 500.)))
    proof = _proof((_stage("stage", (root, left, right)),), physical, _kernel())
    assert not proof["qualified"]
    assert proof["unproven_pairs"] == [{"resource": "cpu.scheduler", "tasks": ["left", "right"]}]


def test_reordering_hbm_scheduler_rejects_the_proof():
    (_runtime, config, physical, _core), _ranks, tasks = fixture(ranks=2)
    original = physical.runtimes["hbm"].core.config
    physical.runtimes["hbm"].core.config = replace(original, scheduler="row_hit_first")
    try:
        proof = _proof((_stage("stage", tasks),), physical, _kernel())
    finally:
        physical.runtimes["hbm"].core.config = original
    assert proof == {"qualified": False, "task_count": len(tasks), "physical_state_advanced": False,
        "reason": "native_physical_owner_may_reorder:hbm"}


def test_dependent_stages_are_ordered_but_independent_stages_are_not():
    (_runtime, _config, physical, _core), _ranks, _tasks = fixture(ranks=2)
    first = _stage("first", (_compute("a", (), "cpu.scheduler"),))
    second = _stage("second", (_compute("b", (), "cpu.scheduler"),), dependencies=("first",))
    assert _proof((first, second), physical, _kernel())["qualified"]
    alone = _stage("second", (_compute("b", (), "cpu.scheduler"),))
    proof = _proof((first, alone), physical, _kernel())
    assert not proof["qualified"] and proof["unproven_pair_count"] == 1


def test_unbounded_scope_and_missing_kernel_fail_closed():
    (_runtime, _config, physical, _core), _ranks, _tasks = fixture(ranks=2)
    stage = _stage("stage", [_compute(f"t{index}", (), f"r{index}") for index in range(4097)])
    assert _proof((stage,), physical, _kernel())["reason"] == "native_issue_order_exceeds_bounded_layer_scope"
    assert _proof((stage,), physical, None)["reason"] == "native_kernel_or_physical_runtime_unavailable"
