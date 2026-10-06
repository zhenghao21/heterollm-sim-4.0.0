import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.memory_types import NandConfig


@pytest.mark.parametrize("value", (1.5, "64", True))
def test_resource_demand_rejects_non_integer_byte_counts(value):
    with pytest.raises(ValueError, match="bytes_moved"):
        ResourceDemand("memory", 1.0, bytes_moved=value)


@pytest.mark.parametrize("field", ("service_ns", "energy_pj", "work_units"))
def test_resource_demand_rejects_non_numeric_metrics(field):
    values = {"service_ns": 1.0, "energy_pj": 0.0, "work_units": 0.0}
    values[field] = "1"
    with pytest.raises(ValueError, match=field):
        ResourceDemand("memory", **values)


@pytest.mark.parametrize("value", (-1, 1.5, "0", True))
def test_task_spec_rejects_invalid_token_indices(value):
    with pytest.raises(ValueError, match="token_index"):
        TaskSpec(
            task_id="task",
            request_id="request",
            name="token",
            category=TaskCategory.OUTPUT,
            token_index=value,
        )


@pytest.mark.parametrize("value", (True, "0", float("nan")))
def test_task_spec_rejects_invalid_earliest_start(value):
    with pytest.raises(ValueError, match="earliest_start_ns"):
        TaskSpec(
            task_id="task",
            request_id="request",
            name="task",
            category=TaskCategory.COMPUTE,
            earliest_start_ns=value,
        )


def test_nand_config_rejects_non_boolean_plane_parallelism():
    with pytest.raises(ValueError, match="planes_independent"):
        NandConfig(planes_independent="false")
