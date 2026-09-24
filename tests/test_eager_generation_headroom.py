"""Eager admission cannot lose its generation reservation during prefill."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from heterollm_sim.serving import _OnlineRuntime


@pytest.mark.parametrize("policy,expected", [("eager", 2056), ("lazy", 2048)])
def test_prefix_reservation_preserves_eager_headroom(policy, expected):
    owner = SimpleNamespace(kv_pages=2056)
    ledger = SimpleNamespace(resize=Mock(return_value=True))
    runtime = SimpleNamespace(
        plan=SimpleNamespace(kv_policy=SimpleNamespace(
            allocation_policy=policy, bytes_per_page=1048576)),
        prompt_cache=SimpleNamespace(ensure_capacity=Mock()), ledger=ledger,
        _reserve_capacity_with_pressure=lambda owner, reserve, protected: reserve())
    assert _OnlineRuntime._reserve_with_pressure(runtime, owner, 2048)
    ledger.resize.assert_called_once_with(owner, expected)


def test_eager_can_grow_beyond_original_reservation():
    owner = SimpleNamespace(kv_pages=2)
    ledger = SimpleNamespace(resize=Mock(return_value=True))
    runtime = SimpleNamespace(
        plan=SimpleNamespace(kv_policy=SimpleNamespace(allocation_policy="eager", bytes_per_page=10)),
        prompt_cache=SimpleNamespace(ensure_capacity=Mock()), ledger=ledger,
        _reserve_capacity_with_pressure=lambda owner, reserve, protected: reserve())
    assert _OnlineRuntime._reserve_with_pressure(runtime, owner, 3)
    ledger.resize.assert_called_once_with(owner, 3)
    runtime.prompt_cache.ensure_capacity.assert_called_once_with(10)
