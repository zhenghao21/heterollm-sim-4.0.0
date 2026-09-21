from pathlib import Path
import sys
from types import SimpleNamespace

ROUND = Path(__file__).resolve().parents[1]
SOURCE = ROUND.parent / "candidate_paired" / "source" / "src"
sys.path.insert(0, str(SOURCE))

from heterollm_sim import planner  # noqa: E402
from heterollm_sim.contracts import ResourceDemand, TaskCategory  # noqa: E402
from heterollm_sim.scalable_serving import TaskExecutionRecord  # noqa: E402
import heterollm_sim.contracts as contracts  # noqa: E402
import heterollm_sim.scalable_serving as scalable_serving  # noqa: E402


EXPECTED_SOURCE_ROOT = (ROUND.parent / "candidate_paired" / "source" / "src").resolve()
assert Path(planner.__file__).resolve().is_relative_to(EXPECTED_SOURCE_ROOT)
assert Path(contracts.__file__).resolve().is_relative_to(EXPECTED_SOURCE_ROOT)
assert Path(scalable_serving.__file__).resolve().is_relative_to(EXPECTED_SOURCE_ROOT)


def _scenario():
    return SimpleNamespace(
        hardware=SimpleNamespace(components=(
            SimpleNamespace(component_id="gpu", kind="gpu"),
            SimpleNamespace(component_id="cpu", kind="cpu"),
        )),
        host_orchestration_profile=SimpleNamespace(cpu_component_id="cpu"),
    )


def _records(*, zero_service_device=False, positive_service_device=False):
    model_gpu = (ResourceDemand("gpu", 5.0),)
    device_gpu = (ResourceDemand("gpu", 1.0),)
    cpu0 = (ResourceDemand("cpu", 2.0),)
    cpu1 = (ResourceDemand("cpu", 3.0),)
    records = [
        TaskExecutionRecord(
            "model", 0.0, 5.0, TaskCategory.COMPUTE, (),
            {"operator_invocation_group_id": "g", "layer_id": "L"}, model_gpu,
        )
    ]
    if zero_service_device or positive_service_device:
        end = 5.0 if zero_service_device else 6.0
        records.append(TaskExecutionRecord(
            "shared", 5.0, end, TaskCategory.SYNCHRONIZATION, ("model",),
            {
                "operator_invocation_group_id": "g",
                "serving_output_stage": "device_output_completion",
                "output_source_component_id": "gpu",
            }, device_gpu if positive_service_device else (),
        ))
    start = 5.0 if zero_service_device else 6.0 if positive_service_device else 5.0
    records.extend((
        TaskExecutionRecord(
            "row0", start, start + 2.0, TaskCategory.COMPUTE,
            ("shared",) if zero_service_device or positive_service_device else ("model",),
            {
                "operator_invocation_group_id": "g",
                "serving_output_stage": "cpu_output_commit",
                "item_index": 0,
                "request_id": "r0",
            }, cpu0,
        ),
        TaskExecutionRecord(
            "row1", start + 2.0, start + 5.0, TaskCategory.COMPUTE, ("row0",),
            {
                "operator_invocation_group_id": "g",
                "serving_output_stage": "cpu_output_commit",
                "item_index": 1,
                "request_id": "r1",
            }, cpu1,
        ),
    ))
    return tuple(records)


def _stages(records):
    return planner._compact_execution_stages(
        _scenario(), records, ({"group_id": "g", "request_ids": ("r0", "r1")},)
    )


def _cpu_output_stages(stages):
    return [stage for stage in stages if stage.get("stage_role") == "cpu_output_commit"]


def test_two_rows_split_without_shared_sync():
    stages, reason = _stages(_records())
    assert reason is None
    rows = _cpu_output_stages(stages)
    assert [row["row_item_index"] for row in rows] == [0, 1]
    assert rows[1]["dependencies"] == (rows[0]["stage_id"],)


def test_zero_service_shared_sync_blocks_row_split():
    stages, reason = _stages(_records(zero_service_device=True))
    assert reason is None
    rows = _cpu_output_stages(stages)
    assert len(rows) == 1
    assert "row_item_index" not in rows[0]
    assert rows[0]["stage_id"] == "g.output_cpu_commit"


def test_positive_service_device_keeps_rows_split():
    stages, reason = _stages(_records(positive_service_device=True))
    assert reason is None
    rows = _cpu_output_stages(stages)
    assert [row["row_item_index"] for row in rows] == [0, 1]
    assert any(stage.get("stage_role") == "device_output_completion" for stage in stages)
