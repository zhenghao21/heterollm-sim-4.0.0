"""The Python source and its strict Cython execution retain Python semantics."""
from collections.abc import Mapping
import importlib.util
from pathlib import Path
import sys
from types import MethodType

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import CompiledGraphLayout, UnifiedEventKernel


@pytest.fixture(scope="module")
def implementations():
    from heterollm_sim import _token_program
    if Path(_token_program.__file__).suffix != ".pyd":
        pytest.skip("same-source native parity requires the local strict Cython extension")
    source = Path(__file__).resolve().parents[1] / "src" / "heterollm_sim" / "_token_program.py"
    name = "heterollm_sim._token_program_python_diagnostic"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module, _token_program
    finally:
        del sys.modules[name]


def bits(value):
    if isinstance(value, float):
        return (type(value).__qualname__, float.hex(value))
    if isinstance(value, Mapping):
        return {key: bits(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(bits(item) for item in value))
    return (type(value).__qualname__, value)


def execute(implementation, tasks):
    kernel = UnifiedEventKernel(capture_physical_details=False)
    # Bind both shared transaction primitives to the selected source too;
    # otherwise the Python frame would still call the compiled delegates.
    kernel._resolve_task_execution = MethodType(implementation.resolve_task_execution, kernel)
    kernel._account_task_execution = MethodType(implementation.account_task_execution, kernel)
    events = []
    implementation.drain_token_program(kernel, tasks, CompiledGraphLayout.compile(tasks),
                                       event_observer=events.append)
    observed = [(event.task.task_id, bits(event.start_ns), bits(event.end_ns),
                 bits(event.dependency_ready_ns), bits(event.effective_ready_ns),
                 bits(event.resource_predecessors), bits(event.resource_lanes)) for event in events]
    state = {name: bits(getattr(kernel, name)) for name in (
        "resource_available", "resource_last_interval", "_resource_lane_available",
        "_resource_lane_last_interval", "completed_end_ns", "completion_leases")}
    state["metrics"] = bits(kernel.metrics)
    return observed, state


@pytest.mark.parametrize("start,duration", [(0, 1), (0., 1), (0, 1.), (-0., -0.),
                                           (0, -0.), (float(2**53), 1), (.1, .2)])
def test_integer_float_types_and_signed_zero_match_same_source(implementations, start, duration):
    first = TaskSpec("a", "request", "a", TaskCategory.COMPUTE,
                     demands=(ResourceDemand("wire", duration),), earliest_start_ns=start)
    second = TaskSpec("b", "request", "b", TaskCategory.COMPUTE,
                      dependencies=(first.task_id,), demands=(ResourceDemand("wire", 0),))
    assert execute(implementations[0], (first, second)) == execute(implementations[1], (first, second))


class ObservedMapping(Mapping):
    def __init__(self, data):
        self.data = data
        self.calls = []

    def __iter__(self):
        self.calls.append(("iter",))
        return iter(self.data)

    def __getitem__(self, key):
        self.calls.append(("getitem", key))
        return self.data[key]

    def __len__(self):
        self.calls.append(("len",))
        return len(self.data)


def test_custom_metadata_mapping_protocol_remains_python(implementations):
    results, protocols = [], []
    for implementation in implementations:
        metadata = ObservedMapping({"authored": "custom mapping"})
        task = TaskSpec("a", "request", "a", TaskCategory.COMPUTE,
                        demands=(ResourceDemand("wire", .1),), metadata=metadata)
        results.append(execute(implementation, (task,)))
        protocols.append(list(metadata.calls))
        assert task.metadata is metadata
    assert results[0] == results[1]
    assert protocols[0] == protocols[1]
    assert protocols[0]
