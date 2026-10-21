"""Internal task accounting preserves every public per-request reduction."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest

from heterollm_sim import _compiled_hbm, _hbm_pack
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import configuration


RESOURCE_FIELDS = ("resource_busy_ns", "resource_bytes", "resource_last_intervals")


@pytest.fixture(params=("python", "compiled"))
def helpers(request, monkeypatch):
    module = _hbm_pack
    if request.param == "python":
        path = Path(__file__).resolve().parents[1] / "src/heterollm_sim/_hbm_pack.py"
        spec = importlib.util.spec_from_file_location("hbm_accumulator_source", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    elif Path(module.__file__).suffix == ".py":
        pytest.skip("optional compiled HBM pack is not installed")
    for name in ("can_capture_resource_columns", "consume_resource_columns"):
        monkeypatch.setattr(_compiled_hbm, name, getattr(module, name))
    return module


def without_resources(result):
    return replace(result, counters={key: value for key, value in result.counters.items()
                                    if key not in RESOURCE_FIELDS})


def assert_bill_equal(actual, expected):
    for field in RESOURCE_FIELDS:
        left, right = getattr(actual, field), getattr(expected, field)
        assert list(left) == list(right), field
        assert left == right, field
        if field == "resource_busy_ns":
            assert [value.hex() for value in left.values()] == [value.hex() for value in right.values()]


def assert_state_equal(left, right):
    assert left._banks == right._banks  # Publish resident controller state.
    assert {key: value for key, value in vars(left).items() if key != "timeline"} == {
        key: value for key, value in vars(right).items() if key != "timeline"}
    # This derived ordering cache belongs only to the public dictionary
    # reducer; omitting that reducer need not warm its private cache.
    assert {key: value for key, value in vars(left.timeline).items() if key != "_metric_key_orders"} == {
        key: value for key, value in vars(right.timeline).items() if key != "_metric_key_orders"}


@pytest.mark.parametrize("mapping", ("bank_interleaved", "row_local"))
def test_singletons_and_mixed_batches_preserve_exact_bills_and_results(helpers, mapping):
    config = configuration(channels=8, address_mapping=mapping,
        interleave_bytes=128 if mapping == "bank_interleaved" else None,
        max_outstanding_requests=1, refresh_policy="all_bank",
        refresh_interval_ns=400.0, refresh_cycle_ns=40.0)
    reference, actual = (DramCore(config, capture_details=False) for _ in range(2))
    # These compare task-local accounting on the numeric controller used by
    # the live physical runtime. The public cold tiny-request dispatcher may
    # deliberately select the separate Python controller, whose diagnostic
    # marker and resource insertion order are not this reducer's contract.
    reference._hbm_numeric_enabled = actual._hbm_numeric_enabled = True
    expected, accumulated = helpers.HBMResourceAccumulator(), helpers.HBMResourceAccumulator()
    span = config.row_bytes * config.bank_count if mapping == "row_local" else 128
    next_id = 0
    for size in (1, 2, 4, 1, 8):
        requests = []
        for index in range(size):
            requests.append(AccessRequest(str(next_id), "read" if next_id % 3 else "write",
                (span * (next_id % config.lane_count) + 3) % (config.effective_capacity_bytes - 8192),
                (65, 2048, 8192)[next_id % 3], 1200.0 + next_id * 93.25))
            next_id += 1
        if size == 1:
            normal = (reference.submit(requests[0]),)
            consumed = (actual.submit(requests[0], _resource_accumulator=accumulated),)
        else:
            normal = reference.submit_many(requests)
            consumed = actual.submit_many(requests, _resource_accumulator=accumulated)
        for result in normal:
            expected.consume_metrics(result.counters)
        assert consumed == tuple(without_resources(result) for result in normal)
        assert_bill_equal(accumulated, expected)
        for field in ("ready_ns", "busy_ns", "last_intervals", "bytes_moved", "lane_available"):
            assert list(getattr(actual.timeline, field)) == list(getattr(reference.timeline, field))
            assert getattr(actual.timeline, field) == getattr(reference.timeline, field)
    assert_state_equal(actual, reference)


def test_numeric_consumption_uses_checkpoint_without_public_metrics_snapshot(helpers, monkeypatch):
    core = DramCore(configuration(channels=8), capture_details=False)
    accumulator = helpers.HBMResourceAccumulator()
    def unexpected(**kwargs):
        raise AssertionError("numeric task accounting copied the public resource dictionaries")
    monkeypatch.setattr(core.timeline, "metrics_snapshot", unexpected)
    core.submit(AccessRequest("single", "read", 3, 65), _resource_accumulator=accumulator)
    core.submit_many((AccessRequest("a", "read", 128, 65), AccessRequest("b", "write", 512, 2048)),
                     _resource_accumulator=accumulator)
    assert accumulator.resource_busy_ns


@pytest.mark.parametrize("failure", ("decline", "exception"))
def test_native_failure_preserves_successful_prefix_and_consumes_fallback_once(helpers, monkeypatch, failure):
    config = configuration(channels=8)
    reference, actual = (DramCore(config, capture_details=False) for _ in range(2))
    reference._hbm_numeric_enabled = actual._hbm_numeric_enabled = True
    requests = (AccessRequest("prefix", "read", 3, 65),
                AccessRequest("stop", "write", 128, 2048),
                AccessRequest("last", "read", 4096, 4096))
    expected, accumulated = helpers.HBMResourceAccumulator(), helpers.HBMResourceAccumulator()
    normal = reference.submit_many(requests) if failure == "decline" else (reference.submit(requests[0]),)
    for result in normal:
        expected.consume_metrics(result.counters)
    original = _compiled_hbm._HBMNumericState.execute
    stopped = False
    def execute(state, request):
        nonlocal stopped
        result = original(state, request)
        if request.request_id == "stop" and not stopped:
            stopped = True
            if failure == "decline":
                return None
            raise RuntimeError("injected numeric failure")
        return result
    monkeypatch.setattr(_compiled_hbm._HBMNumericState, "execute", execute)
    if failure == "decline":
        assert actual.submit_many(requests, _resource_accumulator=accumulated) == tuple(
            without_resources(result) for result in normal)
    else:
        with pytest.raises(RuntimeError, match="injected numeric failure"):
            actual.submit_many(requests, _resource_accumulator=accumulated)
        stopped = False
        with pytest.raises(RuntimeError, match="injected numeric failure"):
            reference.submit(requests[1])
    assert_bill_equal(accumulated, expected)
    assert_state_equal(actual, reference)


@pytest.mark.parametrize("authored", ("integer", "aliases", "custom_timeline", "custom_lane"))
def test_authored_public_state_retains_ordinary_accounting(helpers, monkeypatch, authored):
    class CustomTimeline(ResourceTimeline):
        def metrics_delta(self, before):
            result = super().metrics_delta(before)
            result["custom_observation"] = 23
            return result
    class Calendar(list):
        pass
    config = configuration(channels=8)
    reference, actual = (DramCore(config, capture_details=False) for _ in range(2))
    for core in (reference, actual):
        if authored == "custom_timeline":
            core.timeline = CustomTimeline()
        elif authored == "aliases":
            core.timeline.ready_ns = core.timeline.busy_ns
        elif authored == "custom_lane":
            core.timeline.lane_available["dram:data:0"] = Calendar([0.0])
        else:
            core.timeline.busy_ns["dram:column_command:0:0"] = 2 ** 53 + 1
    def unexpected(*args):
        raise AssertionError("authored state lost its ordinary public arithmetic")
    monkeypatch.setattr(_compiled_hbm, "consume_resource_columns", unexpected)
    requests = (AccessRequest("a", "read", 3, 65), AccessRequest("b", "write", 256, 2048))
    normal = reference.submit_many(requests)
    expected, accumulated = helpers.HBMResourceAccumulator(), helpers.HBMResourceAccumulator()
    for result in normal:
        expected.consume_metrics(result.counters)
    consumed = actual.submit_many(requests, _resource_accumulator=accumulated)
    assert consumed == tuple(without_resources(result) for result in normal)
    assert_bill_equal(accumulated, expected)
    assert_state_equal(actual, reference)


def test_public_dictionary_reordering_is_visible_between_calls(helpers):
    config = configuration(channels=8)
    reference, actual = (DramCore(config, capture_details=False) for _ in range(2))
    warm = AccessRequest("warm", "read", 0, 8192)
    for core in (reference, actual):
        core.submit(warm)
        for field in ("busy_ns", "bytes_moved"):
            values = getattr(core.timeline, field)
            values.update(("unrelated:" + field, 17.0 if field == "busy_ns" else 19) for _ in range(1))
            for key in tuple(values)[::2]:
                values[key] = values.pop(key)
    expected, accumulated = helpers.HBMResourceAccumulator(), helpers.HBMResourceAccumulator()
    request = AccessRequest("changed-order", "read", 0, 8192, 2000.)
    result = reference.submit(request)
    expected.consume_metrics(result.counters)
    assert actual.submit(request, _resource_accumulator=accumulated) == without_resources(result)
    assert_bill_equal(accumulated, expected)


def test_accumulator_keeps_arbitrary_precision_byte_totals(helpers):
    accumulator = helpers.HBMResourceAccumulator()
    for value in (2 ** 80, 2 ** 81, 3):
        accumulator.consume_metrics({"resource_bytes": {"wire": value}})
    assert accumulator.resource_bytes == {"wire": 3 * 2 ** 80 + 3}
