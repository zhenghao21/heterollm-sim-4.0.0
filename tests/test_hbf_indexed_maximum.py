"""Ordered page maxima retain NumPy bits, aliasing and dispatch semantics."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from heterollm_sim import _nand_pack


@pytest.fixture(params=["source", "native"])
def implementation(request):
    if request.param == "native":
        if Path(_nand_pack.__file__).suffix != ".pyd":
            pytest.skip("requires the strict local Cython build")
        return _nand_pack.ordered_indexed_maximum
    path = Path(__file__).resolve().parents[1] / "src" / "heterollm_sim" / "_nand_pack.py"
    spec = importlib.util.spec_from_file_location("hbf_indexed_max_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ordered_indexed_maximum


def assert_same(target, indexes, values, implementation):
    expected = target.copy()
    actual = target.copy()
    with np.errstate(all="ignore"):
        assert np.maximum.at(expected, indexes, values) is None
        assert implementation(actual, indexes, values) is None
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("seed", range(6))
def test_complete_stream_and_equal_signed_zero_bits(implementation, seed):
    random = np.random.default_rng(seed)
    indexes = random.integers(0, 257, 16384, dtype=np.int64)
    values = random.uniform(-1e12, 1e12, len(indexes))
    indexes[-8:] = [1, 1, 2, 2, 3, 3, 4, 4]
    values[-8:] = [0., -0., -0., 0., 7., 7., -np.nextafter(0., 1.), np.nextafter(0., 1.)]
    indexes.flags.writeable = False
    values.flags.writeable = False
    assert_same(np.full(257, -np.inf), indexes, values, implementation)


@pytest.mark.parametrize("values", [[0., -0.], [-0., 0.], [0., -0., 0., -0.],
                                    [-np.nextafter(0., 1.), -0., np.nextafter(0., 1.)]])
def test_finite_ties_store_the_later_operand_bits(implementation, values):
    assert_same(np.array([-np.inf]), np.zeros(len(values), dtype=np.int64),
                np.asarray(values), implementation)


@pytest.mark.parametrize("bits", [0x7ff8000000000001, 0xfff8000000000042,
                                 0x7ff0000000000001, 0xfff0000000000001,
                                 0x7ff0000000000000, 0xfff0000000000000])
def test_nonfinite_payload_uses_original_ufunc(implementation, bits):
    value = np.array([bits], dtype=np.uint64).view(np.float64)[0]
    values = np.array([1., value, 4., -0., 0.])
    assert_same(np.full(2, -np.inf), np.array([0, 0, 0, 1, 1], np.int64), values, implementation)
    assert_same(np.array([value, -np.inf]), np.array([0, 1], np.int64),
                np.array([2., -0.]), implementation)


@pytest.mark.parametrize("case", ["negative", "broadcast", "list", "strided", "float32", "non_native", "unaligned"])
def test_general_numpy_inputs(implementation, case):
    target = np.full(4, -np.inf)
    indexes = np.array([0, 1, 0, 3], dtype=np.int64)
    values = np.array([1., -0., 2., 4.])
    if case == "negative":
        indexes = np.array([-4, -3, -4, -1], dtype=np.int64)
    elif case == "broadcast":
        values = np.array(-0.)
    elif case == "list":
        indexes, values = indexes.tolist(), values.tolist()
    elif case == "strided":
        indexes = np.array([0, 99, 1, 99, 0, 99, 3, 99], dtype=np.int64)[::2]
        values = np.array([1., 99., -0., 99., 2., 99., 4., 99.])[::2]
    elif case == "float32":
        target, values = target.astype(np.float32), values.astype(np.float32)
    elif case == "non_native":
        target, values = target.astype(">f8"), values.astype(">f8")
    elif case == "unaligned":
        array = np.ndarray(4, dtype=np.float64, buffer=bytearray(33), offset=1)
        array[:] = values
        values = array
    assert_same(target, indexes, values, implementation)


@pytest.mark.parametrize("case", ["target_strided", "value_alias", "index_alias", "readonly", "out_of_bounds"])
def test_output_aliases_and_failures_preserve_mutated_bits(implementation, case):
    def fixture():
        target = np.array([0., 1., 2., 3.])
        indexes = np.array([1, 0, 3, 2], dtype=np.int64)
        values = np.array([8., 7., 6., 5.])
        if case == "target_strided":
            target = np.array([0., 99., 1., 99., 2., 99., 3., 99.])[::2]
        elif case == "value_alias":
            values = target[::-1]
        elif case == "index_alias":
            target = np.array([0, 1, 2, 3], dtype=np.int64).view(np.float64)
            indexes = target.view(np.int64)
        elif case == "readonly":
            target.flags.writeable = False
        elif case == "out_of_bounds":
            indexes[-1] = 40
        return target, indexes, values

    expected, actual = fixture(), fixture()
    errors = []
    for function, args in ((np.maximum.at, expected), (implementation, actual)):
        try:
            function(*args)
        except Exception as error:
            errors.append((type(error), str(error)))
        else:
            errors.append(None)
    assert errors[0] == errors[1]
    for before, after in zip(expected, actual):
        assert before.tobytes() == after.tobytes()


@pytest.mark.parametrize("argument", [0, 1, 2])
def test_ndarray_subclass_ufunc_protocol_is_unchanged(implementation, argument):
    class Observed(np.ndarray):
        def __array_finalize__(self, original):
            self.calls = getattr(original, "calls", [])

        def __array_ufunc__(self, ufunc, method, *args, **kwargs):
            self.calls.append((ufunc.__name__, method, args[0].view(np.ndarray).tobytes()))
            plain = tuple(value.view(np.ndarray) if isinstance(value, Observed) else value
                          for value in args)
            return getattr(ufunc, method)(*plain, **kwargs)

    def fixture():
        args = [np.full(4, -np.inf), np.array([0, 1, 0, 3], np.int64),
                np.array([1., -0., 2., 4.])]
        args[argument] = args[argument].view(Observed)
        return args

    expected, actual = fixture(), fixture()
    np.maximum.at(*expected)
    implementation(*actual)
    assert expected[argument].calls == actual[argument].calls
    assert actual[argument].calls[0][0:2] == ("maximum", "at")
    assert expected[0].tobytes() == actual[0].tobytes()


def test_empty_stream_preserves_target(implementation):
    assert_same(np.array([-0., np.nextafter(0., 1.)]), np.array([], np.int64),
                np.array([], np.float64), implementation)
