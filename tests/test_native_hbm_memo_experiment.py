"""The optional memo replays complete numeric transitions without approximations."""
from concurrent.futures import ThreadPoolExecutor
import math

import numpy as np
import pytest

from heterollm_sim import _hbm_numeric as numeric
from heterollm_sim import native_hbm_memo_experiment as memos
from heterollm_sim.native_hbm_memo_experiment import hbm_numeric_memo_experiment
from heterollm_sim.native_hbm_memo_experiment import _ACTIVE


def inputs():
    g = np.arange(13, dtype=np.int64)
    g[:4] = (4, 32, 1, 32)
    g[9] = 32
    t = np.arange(31, dtype=np.float64)
    arrays = tuple(np.zeros((4, 4), dtype=np.float64) for _ in range(14))
    return g, t, arrays


def large_inputs():
    g, t, _ = inputs()
    return g, t, tuple(np.zeros((128, 128), dtype=np.float64) for _ in range(14))


@pytest.fixture
def native(monkeypatch):
    calls = []

    def execute(first, stop, arrival, read, g, t, arrays, scratch):
        calls.append((first, stop, arrival, read))
        for index, array in enumerate(arrays):
            array += index + 1 + first
        return arrival + float(stop), 1, 2, 3, np.arange(7, dtype=np.float64)

    monkeypatch.setattr(numeric, "_run_hbm_numeric", execute)
    return calls, execute


def call(values, *, first=0, arrival=1.0):
    g, t, arrays = values
    # Unit-test the cache algebra with a fake pure transition. The production
    # bridge deliberately declines replaced kernels; its tests below use the
    # real native function and complete packed geometry.
    active = _ACTIVE.get()
    if active is None:
        return numeric._run_hbm_numeric(first, 8192, arrival, True, g, t, arrays, object())
    return active.execute(numeric._run_hbm_numeric, first, 8192, arrival, True, g, t, arrays, object())


def test_complete_post_state_and_mutable_result_are_independent_on_hits(native):
    calls, original = native
    with hbm_numeric_memo_experiment(min_request_bytes=0) as memo:
        values = inputs()
        result = call(values)
        expected = [array.tobytes() for array in values[2]]
        result[4][0] = 999
        repeated = inputs()
        replayed = call(repeated)
        assert len(calls) == 1 and memo.stats["hits"] == 1
        assert [array.tobytes() for array in repeated[2]] == expected
        assert replayed[4][0] == 0
        assert memo.stats["resident_bytes"] <= memo.max_bytes
    assert numeric._run_hbm_numeric is original
    assert memo.stats["entries"] == memo.stats["resident_bytes"] == 0


def test_every_float_bit_and_every_array_is_part_of_the_key(native):
    calls, _ = native
    with hbm_numeric_memo_experiment(min_request_bytes=0) as memo:
        call(inputs())
        call(inputs(), arrival=math.nextafter(1.0, 2.0))
        changed_t = inputs()
        changed_t[1][0] = math.nextafter(0.0, 1.0)
        call(changed_t)
        changed_state = inputs()
        changed_state[2][-1][0] = -0.0  # Numerically equal, different IEEE bits.
        call(changed_state)
        changed_g = inputs()
        changed_g[0][0] += 1
        call(changed_g)
        assert len(calls) == 5 and memo.stats["hits"] == 0


def test_lossless_frames_are_deterministic_and_preserve_every_byte():
    # Include sign bits, NaN payloads, and arbitrary non-floating byte patterns.
    for contents in (b"", bytes(range(256)), bytes(range(256)) * 64):
        frame = memos._encode_image(contents)
        assert frame == memos._encode_image(contents)
        assert frame[0] == memos.IMAGE_FORMAT_VERSION
        assert frame[1] == (memos._ZLIB_CODEC if len(contents) >= 4096 else memos._RAW_CODEC)
        assert memos._decode_image(frame) == contents
    with pytest.raises(ValueError, match="length mismatch"):
        memos._decode_image((memos.IMAGE_FORMAT_VERSION, memos._RAW_CODEC, 2, b"a"))


def test_compressed_input_keys_keep_every_array_bit_distinct(native):
    calls, _ = native
    with hbm_numeric_memo_experiment(min_request_bytes=0) as memo:
        call(large_inputs())
        for index in range(16):
            values = large_inputs()
            (values[:2] + values[2])[index].view(np.uint8).reshape(-1)[-1] ^= 1
            call(values)
        call(large_inputs())
        assert len(calls) == 17 and memo.stats["misses"] == 17 and memo.stats["hits"] == 1
        assert memo.stats["raw_resident_image_bytes"] > memo.stats["stored_resident_image_bytes"]
        assert all(type(value) is int for name, value in memo.stats.items()
                   if "image" in name)


def test_lru_never_retains_more_than_its_configured_limit(native):
    calls, _ = native
    with hbm_numeric_memo_experiment(min_request_bytes=0) as initial:
        call(inputs())
        one_entry = initial.stats["resident_bytes"]
    with hbm_numeric_memo_experiment(max_bytes=one_entry + 1, min_request_bytes=0) as memo:
        call(inputs(), first=0)
        call(inputs(), first=1)
        call(inputs(), first=0)
        assert memo.stats["evictions"] == 2 and memo.stats["entries"] == 1
        assert memo.stats["resident_bytes"] <= one_entry + 1


def test_lru_charges_compressed_images_and_their_actual_containers(native):
    with hbm_numeric_memo_experiment(min_request_bytes=0) as initial:
        call(large_inputs())
        one_entry = initial.stats["resident_bytes"]
        assert initial.stats["raw_resident_image_bytes"] > 20 * one_entry
    with hbm_numeric_memo_experiment(max_bytes=2 * one_entry + 1024, min_request_bytes=0) as memo:
        call(large_inputs(), first=0)
        call(large_inputs(), first=1)
        assert memo.stats["entries"] == 2 and memo.stats["evictions"] == 0
        call(large_inputs(), first=0)  # Keep zero, evict one when two arrives.
        call(large_inputs(), first=2)
        assert memo.stats["entries"] == 2 and memo.stats["evictions"] == 1
        assert memo.stats["hits"] == 1
        call(large_inputs(), first=1)
        assert memo.stats["misses"] == 4 and memo.stats["evictions"] == 2
        assert memo.stats["raw_resident_image_bytes"] > memo.max_bytes
        assert memo.stats["resident_bytes"] <= memo.max_bytes
        assert memo.stats["resident_payload_bytes"] == sum(entry[2] for entry in memo._entries.values())
        assert memo.stats["raw_resident_image_bytes"] == sum(entry[3] for entry in memo._entries.values())
        assert memo.stats["stored_resident_image_bytes"] == sum(entry[4] for entry in memo._entries.values())
        assert memo.stats["resident_bytes"] == (memo.stats["resident_payload_bytes"]
            + memos.sys.getsizeof(memo._entries) - memos._EMPTY_LRU_BYTES)
    assert memo.stats["raw_resident_image_bytes"] == memo.stats["stored_resident_image_bytes"] == 0


def test_other_threads_bypass_this_context_and_nested_contexts_are_isolated(native):
    calls, _ = native
    with hbm_numeric_memo_experiment(min_request_bytes=0) as outer:
        call(inputs())
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(call, inputs()).result()
        with hbm_numeric_memo_experiment(min_request_bytes=0) as inner:
            call(inputs())
            call(inputs())
            assert inner.stats["hits"] == 1
        call(inputs())
        assert outer.stats["hits"] == 1 and len(calls) == 3


def test_control_checks_are_not_reused_from_the_cached_call(native):
    denied = False

    def check():
        if denied:
            raise TimeoutError("current deadline")

    with hbm_numeric_memo_experiment(min_request_bytes=0, check=check):
        call(inputs())
        denied = True
        fresh = inputs()
        with pytest.raises(TimeoutError, match="current deadline"):
            call(fresh)
        assert not any(array.any() for array in fresh[2])


def test_failures_are_not_cached_and_original_exception_is_preserved(monkeypatch):
    calls = []

    def failed(*args):
        calls.append(1)
        raise OverflowError("native safe integer limit")

    monkeypatch.setattr(numeric, "_run_hbm_numeric", failed)
    with hbm_numeric_memo_experiment(min_request_bytes=0) as memo:
        for _ in range(2):
            with pytest.raises(OverflowError, match="native safe integer limit"):
                call(inputs())
        assert len(calls) == 2 and memo.stats["entries"] == 0


def test_aliases_decline_the_memo_and_execute_the_original(native):
    calls, _ = native
    values = inputs()
    values = values[:2] + ((values[2][0],) * 14,)
    with hbm_numeric_memo_experiment(min_request_bytes=0) as memo:
        call(values)
        assert memo.stats["declined"] == 1 and len(calls) == 1


def test_real_numeric_hit_keeps_native_publish_and_does_not_bypass_hook_guard():
    from heterollm_sim import _compiled_hbm
    from heterollm_sim.dram_core import DramCore
    from heterollm_sim.memory_types import AccessRequest
    from test_hbm_compiled_control import configuration
    from test_token_owned_history import bits

    request = AccessRequest("same-native-transfer", "read", 0, 256 * 1024, 0.0)
    cores = [DramCore(configuration(), capture_details=False) for _ in range(3)]
    expected = _compiled_hbm.execute_hbm_numeric(cores[0], request)
    assert expected is not None and _compiled_hbm._has_native_timeline_hooks()
    with hbm_numeric_memo_experiment() as memo:
        assert _compiled_hbm._has_native_timeline_hooks()
        outputs = [_compiled_hbm.execute_hbm_numeric(core, request) for core in cores[1:]]
        assert memo.stats["hits"] == 1 and memo.stats["misses"] == 1
        assert memo.stats["stored_resident_image_bytes"] < memo.stats["raw_resident_image_bytes"]
        assert any(image[1] == memos._ZLIB_CODEC for entry in memo._entries.values() for image in entry[0])
    assert _compiled_hbm._has_native_timeline_hooks()
    assert bits(outputs) == bits([expected, expected])
    packed = [_compiled_hbm._pack_hbm_numeric(core, request) for core in cores]
    images = [tuple((array.dtype.str, array.shape, array.tobytes()) for array in state.arrays)
              for state in packed]
    assert images[0] == images[1] == images[2]
    for core in cores:
        core.timeline.metrics_snapshot()
    for actual in cores[1:]:
        assert bits(actual._hbm_control_state) == bits(cores[0]._hbm_control_state)
        assert bits(actual._banks) == bits(cores[0]._banks)
        for field in ("ready_ns", "busy_ns", "bytes_moved", "directions", "last_intervals", "lane_available"):
            assert bits(getattr(actual.timeline, field)) == bits(getattr(cores[0].timeline, field))
