"""Link eligibility preserves Python max order, identity and mapping hooks."""
from collections import UserDict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

from heterollm_sim._io_execution import _PhysicalIO
from heterollm_sim.memory_transfer import ResourceTimeline
from test_hbf_staging_runtime import setup


@pytest.fixture(scope="module", params=("compiled", "source"))
def io_type(request):
    if request.param == "compiled":
        return _PhysicalIO
    path = Path(__file__).resolve().parents[1] / "src/heterollm_sim/_io_execution.py"
    spec = spec_from_file_location("heterollm_sim._io_execution_protocol_source", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._PhysicalIO


class _Clock(float):
    def __new__(cls, value, name, comparisons):
        item = super().__new__(cls, value)
        item.name, item.comparisons = name, comparisons
        return item

    def __gt__(self, other):
        self.comparisons.append((self.name, getattr(other, "name", "plain")))
        return super().__gt__(other)


class _ProtocolDict(UserDict):
    def __init__(self):
        self.reads = self.writes = 0
        super().__init__()

    def __getitem__(self, key):
        self.reads += 1
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        self.writes += 1
        return super().__setitem__(key, value)


@pytest.mark.parametrize("custom_mapping", [False, True])
@pytest.mark.parametrize("size", [0, 256])
@pytest.mark.parametrize("arrival,ready", [
    (0., (-0., 0.)),
    (-0., (0., 0.)),
    (3., (float("nan"), 5.)),
    (3., (5., float("nan"))),
    (7., (7., 7.)),
    (float("nan"), (5., 9.)),
])
def test_link_start_matches_ordered_builtin_max(io_type, custom_mapping, size, arrival, ready):
    comparisons, actual_calls = [], []
    current = _Clock(arrival, "arrival", comparisons)
    values = tuple(_Clock(value, name, comparisons)
                   for value, name in zip(ready, ("wire", "credit")))
    expected = max((current, *values)) if size else current
    expected_comparisons = list(comparisons)
    comparisons.clear()

    class Timeline(ResourceTimeline):
        armed = False

        def available_ns(self, resource):
            if self.armed:
                if resource == "to_host":
                    return values[0]
                if resource == "to_host.inflight":
                    return values[1]
            return super().available_ns(resource)

        def reserve(self, resource, earliest, duration, **kwargs):
            actual_calls.append(earliest)
            return super().reserve(resource, earliest, duration, **kwargs)

    timeline = Timeline()
    custom = []
    if custom_mapping:
        for field in ("lane_available", "ready_ns", "busy_ns", "bytes_moved",
                      "last_intervals", "directions", "lane_directions"):
            mapping = _ProtocolDict()
            setattr(timeline, field, mapping)
            custom.append(mapping)
    _, binding = setup()
    records = []
    io = io_type(SimpleNamespace(timeline=timeline, capture_details=False),
                 "flash", {}, binding, records)
    # Keep physical lanes valid while the authored eligibility method returns
    # probe values. LinkService declares credit capacity separately from wire.
    timeline.lane_available["to_host"] = [0.]
    io._route("to_host", size)
    for mapping in custom:
        mapping.reads = mapping.writes = 0
    timeline.armed = True
    if arrival != arrival and size:
        with pytest.raises(ValueError, match="finite"):
            io.transfer("to_host", size, current)
        assert not records
    else:
        io.transfer("to_host", size, current)
        assert records[0][-1].logical_bytes == size
        assert records[0][-1].transfer_bytes == size
    assert comparisons == expected_comparisons
    assert bool(actual_calls) == bool(size)
    assert all(value is expected for value in actual_calls)
    if custom_mapping and size and arrival == arrival:
        assert sum(mapping.reads for mapping in custom) > 0
        assert sum(mapping.writes for mapping in custom) > 0
