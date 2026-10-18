"""Native FIFO conversion retains public command order and scalar values."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from heterollm_sim import _nand_pack
from heterollm_sim.memory_types import Operation


@pytest.fixture(params=["source", "native"])
def implementation(request):
    if request.param == "native":
        if Path(_nand_pack.__file__).suffix != ".pyd":
            pytest.skip("requires local strict Cython build")
        return _nand_pack
    path = Path(__file__).resolve().parents[1] / "src" / "heterollm_sim" / "_nand_pack.py"
    spec = importlib.util.spec_from_file_location("nand_pack_source_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_roundtrip_keeps_order_write_kinds_and_untouched_host(implementation):
    old = [(7.25, Operation.WRITE), (-0., Operation.READ), (2**42 + .125, Operation.READ)]
    queues = {2: old, 0: [(1., Operation.WRITE)], 17: [(99., Operation.READ)]}
    acceptances = {2: 3.25, 17: 99.}
    values = implementation.pack_host_admission(queues, acceptances, .5, 4, 2, Operation.READ)
    times, operations, counts, frontiers = values
    assert times.shape == operations.shape == (4, 4)
    assert counts.tolist() == [1, 0, 3, 0]
    assert frontiers.tolist() == [.5, .5, 3.25, .5]
    assert operations[2, :3].tolist() == [2, 1, 1]
    assert times[2, 1].tobytes() == np.float64(-0.).tobytes()
    touched = np.array([False, True, True, False])
    implementation.publish_host_admission(queues, acceptances, *values, touched,
                                           Operation.READ, Operation.WRITE)
    assert list(queues) == [2, 0, 17, 1]
    assert list(acceptances) == [2, 17, 1]
    assert queues[2] == old and queues[2] is not old
    assert queues[2][1][0].hex() == (-0.).hex()
    assert all(type(end) is float for end, _ in queues[2])
    assert queues[1] == [] and queues[17] == [(99., Operation.READ)]
    assert queues[0] == [(1., Operation.WRITE)]


class ObservedDict(dict):
    def __init__(self, data):
        super().__init__(data)
        self.calls = []

    def get(self, key, default=None):
        self.calls.append(("get", key))
        return super().get(key, default)

    def __setitem__(self, key, value):
        self.calls.append(("set", key))
        return super().__setitem__(key, value)


def test_conversion_honors_custom_mapping_protocol(implementation):
    queues = ObservedDict({0: [(1, Operation.READ)]})
    acceptances = ObservedDict({})
    values = implementation.pack_host_admission(queues, acceptances, 0., 2, 2, Operation.READ)
    implementation.publish_host_admission(queues, acceptances, *values,
        np.array([True, False]), Operation.READ, Operation.WRITE)
    assert queues.calls == [("get", 0), ("get", 1), ("set", 0)]
    assert acceptances.calls == [("get", 0), ("get", 1), ("set", 0)]
    assert type(queues[0][0][0]) is float


@pytest.mark.parametrize("base,per_channel,hosts,unit_count,bank_ids,unit_ids,expected", [
    (8, 2, 4, 24, [2, 10, 18], [], [1]),  # READ hits need no array touch.
    (8, 2, 4, 24, [], [6, 14, 22], [3]),  # Independent-plane cold writes.
    (8, 2, 4, 8, [], [6], [3]),           # Shared-plane cold writes.
    (8, 2, 4, 24, [2, 10], [4], [1, 2]),
    (8, 2, 2, 8, [6, 14], [], [1]),
    (8, 2, 4, 24, [], [], []),
    (32, 2, 16, 96, [32, 95], [64], [0, 15]),
])
def test_host_mask_covers_read_banks_and_shared_or_independent_write_units(
        implementation, base, per_channel, hosts, unit_count, bank_ids, unit_ids, expected):
    banks = np.zeros(base * 3, dtype=np.bool_)
    units = np.zeros(unit_count, dtype=np.bool_)
    banks[bank_ids] = True
    units[unit_ids] = True
    before_banks, before_units = banks.copy(), units.copy()
    result = implementation.host_admission_mask(banks, units, base, per_channel, hosts)
    assert result.dtype == np.bool_ and result.shape == (hosts,)
    assert np.flatnonzero(result).tolist() == expected
    np.testing.assert_array_equal(banks, before_banks)
    np.testing.assert_array_equal(units, before_units)


@pytest.mark.parametrize("value", [[1.], (1.,), [1., 2.]])
def test_malformed_fifo_time_keeps_numpy_scalar_assignment_error(implementation, value):
    original = np.zeros((1, 2), dtype=np.float64)
    with pytest.raises(ValueError) as expected:
        original[0, 0] = value
    with pytest.raises(type(expected.value), match="setting an array element with a sequence"):
        implementation.pack_host_admission({0: [(value, Operation.READ)]}, {},
                                            0., 1, 2, Operation.READ)
