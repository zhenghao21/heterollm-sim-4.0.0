"""Serial IEEE addition remains exact when entering a wider ULP interval."""
import math

import pytest

from heterollm_sim.dram_core import _repeat_add as python_repeat
from heterollm_sim._dram_numeric import _repeat_add as native_repeat


@pytest.mark.parametrize("exponent", [-100, -1, 0, 20, 100, 900])
@pytest.mark.parametrize("half_steps", [1, 3, 5, 7, 31])
@pytest.mark.parametrize("count", [0, 1, 2, 3, 10, 1024])
def test_ties_after_binade_crossing(exponent, half_steps, count):
    boundary = math.ldexp(2.0, exponent)
    initial = math.nextafter(boundary, 0.0)
    increment = half_steps * math.ldexp(1.0, exponent - 52)
    expected = initial
    for _ in range(count):
        expected += increment
    assert python_repeat(initial, increment, count).hex() == expected.hex()
    assert native_repeat(initial, increment, count).hex() == expected.hex()


@pytest.mark.parametrize("initial,increment", [(0., .512), (1e9 + .1, .512),
    (0., math.ulp(0.)), (math.ulp(0.), 3 * math.ulp(0.)),
    (math.ldexp(1., 1023), math.ldexp(1., 970))])
def test_repeated_addition_normal_subnormal_and_stagnation(initial, increment):
    expected = initial
    for _ in range(1001):
        expected += increment
    assert python_repeat(initial, increment, 1001).hex() == expected.hex()
    assert native_repeat(initial, increment, 1001).hex() == expected.hex()
