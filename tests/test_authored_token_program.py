"""Resident physical bindings keep the exact authored graph and live payload."""

from dataclasses import replace
from copy import copy, deepcopy
import gc
import weakref

import pytest

from heterollm_sim._token_binding import AuthoredTokenProgram
from heterollm_sim.contracts import (
    ResourceDemand, TaskCategory, _PreparedExecutionStage, _PreparedExecutionTask,
)
from heterollm_sim.planner import (
    _PreparedExecutionStageRows, _PREPARED_EXECUTION_STAGE_TOKEN, _trusted_execution_stages,
)
from heterollm_sim.serving import (
    _ExecutionStageMetadataCache, _OnlineRuntime, _live_task_metadata,
    _declares_explicit_task_dag, _execution_stages_from_metadata,
)


def stages():
    prefix = _PreparedExecutionTask("host", (), ("request",),
        (ResourceDemand("cpu", 1.0),), metadata={"runtime_phase": "host"})
    read = _PreparedExecutionTask("read", (), ("request",),
        (ResourceDemand("memory", 0.0, 128),), category=TaskCategory.MEMORY,
        metadata={"physical_memory_config": {"preset": "placeholder"},
                  "memory_accesses": ({"offset_bytes": 128, "allocation_generation": 0},)})
    compute = _PreparedExecutionTask("compute", ("read",), ("request",),
        (ResourceDemand("gpu", 2.0),), category=TaskCategory.COMPUTE)
    tail = _PreparedExecutionTask("tail", (), ("request",), (ResourceDemand("cpu", 3.0),))
    return (
        _PreparedExecutionStage("prefix", 0, (), ("request",), "cpu", 1.0, (prefix,)),
        _PreparedExecutionStage("body", 1, ("prefix",), ("request",), "gpu", 2.0, (read, compute)),
        _PreparedExecutionStage("suffix", 2, ("body",), ("request",), "cpu", 3.0, (tail,)),
    )


def metadata(rows):
    return {"execution_stages": _PreparedExecutionStageRows((), rows,
        _token=_PREPARED_EXECUTION_STAGE_TOKEN)}


def test_prepared_capsule_is_acyclic_and_copies_cannot_claim_original_ownership():
    def handoff():
        rows = stages()
        reference = weakref.ref(rows[0].execution_tasks[0])
        wrapper = metadata(rows)["execution_stages"]
        return wrapper, reference
    enabled = gc.isenabled()
    gc.disable()
    try:
        wrapper, reference = handoff()
        assert _trusted_execution_stages(wrapper)
        assert _trusted_execution_stages(copy(wrapper)) is None
        assert _trusted_execution_stages(deepcopy(wrapper)) is None
        cloned = _PreparedExecutionStageRows(wrapper)
        cloned._prepared_capsule = wrapper._prepared_capsule
        assert _trusted_execution_stages(cloned) is None
        del cloned, wrapper
        assert reference() is None, "the complete native DAG must retire without cyclic-GC scanning"
    finally:
        if enabled:
            gc.enable()


def test_program_binding_preserves_original_names_dependencies_and_metadata():
    rows = stages()
    ready = {"prefix": 0, "body": 5.0, "suffix": 10.0}
    namespace = "serving.online.00000042"
    program = AuthoredTokenProgram.compile(rows, _OnlineRuntime._stage_task_specs)
    expected, _source = _OnlineRuntime._stage_task_specs(rows, ready, namespace)
    assert program.bind(rows, ready, namespace, _live_task_metadata) == expected
    assert program.compiled.dependency_positions == ((), (0,), (1,), (2,))


def test_reused_program_binds_current_payload_readiness_and_generation():
    first = stages()
    cache = _ExecutionStageMetadataCache()
    program = cache.authored_token_program(metadata(first), first, _OnlineRuntime._stage_task_specs)
    body = first[1]
    read = replace(body.execution_tasks[0],
        metadata={"physical_memory_config": {"preset": "current"},
                  "memory_accesses": ({"offset_bytes": 256, "allocation_generation": 7},)},
        demands=(ResourceDemand("memory", 4.0, 256),))
    second = (first[0], replace(body, execution_tasks=(read, body.execution_tasks[1])), first[2])
    assert cache.authored_token_program(metadata(second), second,
        _OnlineRuntime._stage_task_specs) is program
    ready = {"prefix": 30.0, "body": 35.0, "suffix": 40.0}
    bound = program.bind(second, ready, "serving.online.00000043", _live_task_metadata)
    expected, _source = _OnlineRuntime._stage_task_specs(second, ready, "serving.online.00000043")
    assert bound == expected
    assert bound[1].metadata["memory_accesses"][0]["allocation_generation"] == 7
    assert bound[1].demands[0].bytes_moved == 256


def test_dependency_or_resource_shape_change_compiles_a_distinct_program():
    first = stages()
    cache = _ExecutionStageMetadataCache()
    original = cache.authored_token_program(metadata(first), first, _OnlineRuntime._stage_task_specs)
    body = first[1]
    compute = replace(body.execution_tasks[1], dependencies=(), demands=(ResourceDemand("other_gpu", 2.0),))
    second = (first[0], replace(body, execution_tasks=(body.execution_tasks[0], compute)), first[2])
    changed = cache.authored_token_program(metadata(second), second, _OnlineRuntime._stage_task_specs)
    assert changed is not original
    assert changed.compiled.dependency_positions == ((), (0,), (0,), (1, 2))


def test_plain_custom_stage_metadata_does_not_enter_trusted_program_path():
    rows = stages()
    assert _ExecutionStageMetadataCache().authored_token_program(
        {"execution_stages": tuple(rows)}, rows, _OnlineRuntime._stage_task_specs) is None


@pytest.mark.parametrize("ready", [True, -1.0, float("inf"), float("nan")])
def test_invalid_live_release_times_fail_before_binding(ready):
    rows = stages()
    program = AuthoredTokenProgram.compile(rows, _OnlineRuntime._stage_task_specs)
    with pytest.raises(ValueError):
        program.bind(rows, {"prefix": ready, "body": 0.0, "suffix": 0.0},
            "serving.online.00000044", _live_task_metadata)
