"""The explicit memo bridge retains native ownership only for controlled code."""
from copy import deepcopy

import pytest

from heterollm_sim import _compiled_hbm, _hbm_numeric, execution_control
from heterollm_sim import native_hbm_memo_experiment as memos
from heterollm_sim.dram_core import DramCore
from heterollm_sim.execution_control import ExecutionControl, NativeFastWallBudgetError
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import configuration


def request():
    return AccessRequest("same", "read", 0, 256 * 1024, 0.0)


def execute():
    return _compiled_hbm.execute_hbm_numeric(DramCore(configuration(), capture_details=False), request())


@pytest.mark.parametrize("change", ["cache_type", "check", "control_callback", "execute", "key_helper",
                                    "encode_helper", "decode_helper", "image_sizes_helper",
                                    "zlib_compress", "zlib_decompress", "zlib_module", "image_format",
                                    "numpy_copy", "bridge", "executor"])
def test_unknown_memo_code_or_callback_cannot_claim_native_ownership(monkeypatch, change):
    with memos.hbm_numeric_memo_experiment() as memo:
        assert _compiled_hbm._has_native_timeline_hooks()
        token = None
        if change == "cache_type":
            token = memos._ACTIVE.set(object())
        elif change == "check":
            memo.check = lambda: None
        elif change == "control_callback":
            memo.check = ExecutionControl(cancellation_callback=lambda: False).raise_if_cancelled
        elif change == "execute":
            monkeypatch.setattr(memos.NumericMemoExperiment, "execute", lambda *args, **kwargs: None)
        elif change == "key_helper":
            monkeypatch.setattr(memos, "_array_key", lambda *args: None)
        elif change == "encode_helper":
            monkeypatch.setattr(memos, "_encode_image", lambda *args: None)
        elif change == "decode_helper":
            monkeypatch.setattr(memos, "_decode_image", lambda *args: None)
        elif change == "image_sizes_helper":
            monkeypatch.setattr(memos, "_image_sizes", lambda *args: None)
        elif change == "zlib_compress":
            monkeypatch.setattr(memos.zlib, "compress", lambda *args, **kwargs: b"")
        elif change == "zlib_decompress":
            monkeypatch.setattr(memos.zlib, "decompress", lambda *args, **kwargs: b"")
        elif change == "zlib_module":
            monkeypatch.setattr(memos, "zlib", object())
        elif change == "image_format":
            monkeypatch.setattr(memos, "IMAGE_FORMAT_VERSION", 2)
        elif change == "numpy_copy":
            monkeypatch.setattr(memos.np, "copyto", lambda *args, **kwargs: None)
        elif change == "bridge":
            monkeypatch.setattr(_compiled_hbm, "_numeric_memo_execution_policy", lambda: ("foreign",))
        elif change == "executor":
            monkeypatch.setattr(_hbm_numeric, "_CHANNEL_EXECUTOR", object())
        try:
            assert not _compiled_hbm._has_native_timeline_hooks()
        finally:
            if token is not None:
                memos._ACTIVE.reset(token)


@pytest.mark.parametrize("hook", ["_run_hbm_numeric", "_run_hbm_channel_range"])
def test_changed_native_numeric_function_executes_normally_instead_of_hitting_old_memo(monkeypatch, hook):
    original = getattr(_hbm_numeric, hook)
    calls = []

    def changed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    with memos.hbm_numeric_memo_experiment() as memo:
        expected = execute()
        assert memo.stats["misses"] == 1
        monkeypatch.setattr(_hbm_numeric, hook, changed)
        assert execute() == expected
        assert calls and memo.stats["hits"] == 0 and memo.stats["calls"] == 1


def test_replaced_policy_helper_is_not_invoked_by_the_bridge(monkeypatch):
    calls = []

    def foreign():
        calls.append(1)
        return ("pretend-native",)

    with memos.hbm_numeric_memo_experiment() as memo:
        expected = execute()
        monkeypatch.setattr(_compiled_hbm, "_numeric_memo_execution_policy", foreign)
        assert execute() == expected
        assert not calls and memo.stats["hits"] == 0


@pytest.mark.parametrize("hook", ["_encode_image", "_decode_image"])
def test_unknown_image_helpers_are_not_invoked_by_the_native_bridge(monkeypatch, hook):
    calls = []

    def foreign(*args):
        calls.append(1)
        raise AssertionError("unknown memo image helper must not execute")

    with memos.hbm_numeric_memo_experiment() as memo:
        expected = execute()
        monkeypatch.setattr(memos, hook, foreign)
        assert execute() == expected
        assert not calls and memo.stats["hits"] == 0 and memo.stats["calls"] == 1


def test_worker_grouping_is_part_of_the_exact_key(monkeypatch):
    with memos.hbm_numeric_memo_experiment() as memo:
        monkeypatch.setattr(_hbm_numeric, "_CHANNEL_WORKERS", 2)
        first = execute()
        monkeypatch.setattr(_hbm_numeric, "_CHANNEL_WORKERS", 1)
        second = execute()
        monkeypatch.setattr(_hbm_numeric, "_CHANNEL_WORKERS", 2)
        repeated = execute()
        assert first == second == repeated
        assert memo.stats["misses"] == 2 and memo.stats["hits"] == 1


def test_trusted_deadline_after_native_write_restores_sparse_state_and_foreign_resources(monkeypatch):
    from heterollm_sim.data_motion import PhysicalRuntimeContext

    runtime = PhysicalRuntimeContext(configuration(channels=8, interleave_bytes=128), "hbm", capture_details=False)
    core = runtime.runtimes["hbm"].core
    runtime.timeline.reserve("unrelated-owner:lane", 111.0, 2.0)
    core.submit(AccessRequest("warm", "read", 0, 4096, 0.0))
    packed = core._hbm_numeric_state
    small = AccessRequest("small", "write", 3, 65, 1500.0)
    assert packed.request_selection(small) is not None
    before_arrays = tuple(array.tobytes() for array in packed.arrays)
    before_timeline = deepcopy({name: getattr(runtime.timeline, name) for name in
        ("ready_ns", "busy_ns", "bytes_moved", "directions", "last_intervals", "lane_available")})
    clock = iter((0.0, 2.0))  # Native current-deadline check, after the numeric miss mutates arrays.
    monkeypatch.setattr(execution_control, "perf_counter", lambda: next(clock))
    control = ExecutionControl(deadline_monotonic=1.0)
    with memos.hbm_numeric_memo_experiment(min_request_bytes=0, check=control.raise_if_cancelled) as memo:
        assert _compiled_hbm._has_native_timeline_hooks(packed)
        token = runtime.begin_transaction(owners=("hbm",), _owned_timeline=True)
        assert "_owned_transaction" in token["timeline"]
        with pytest.raises(NativeFastWallBudgetError):
            core.submit(small)
        runtime.restore(token)
        assert memo.stats["entries"] == 0
        assert tuple(array.tobytes() for array in packed.arrays) == before_arrays
        assert not any(key.startswith("unrelated-owner:")
                       for key in token["timeline"]["_owned_transaction"].entries["ready_ns"])
    for name, expected in before_timeline.items():
        assert getattr(runtime.timeline, name) == expected


def test_trusted_deadline_after_memo_copy_restores_every_array_with_sparse_undo(monkeypatch):
    from heterollm_sim.data_motion import PhysicalRuntimeContext

    runtime = PhysicalRuntimeContext(configuration(channels=8, interleave_bytes=128), "hbm", capture_details=False)
    core = runtime.runtimes["hbm"].core
    runtime.timeline.reserve("foreign", 13.0, 1.0)
    core.submit(AccessRequest("warm", "read", 0, 4096, 0.0))
    small = AccessRequest("small", "write", 3, 65, 1500.0)
    packed = core._hbm_numeric_state
    before = tuple(array.tobytes() for array in packed.arrays)
    with memos.hbm_numeric_memo_experiment(min_request_bytes=0) as memo:
        seed = runtime.begin_transaction(owners=("hbm",), _owned_timeline=True)
        core.submit(small)
        runtime.restore(seed)
        assert memo.stats["misses"] == 1 and memo.stats["entries"] == 1
        clock = iter((0.0, 0.0, 2.0))
        monkeypatch.setattr(execution_control, "perf_counter", lambda: next(clock))
        memo.check = ExecutionControl(deadline_monotonic=1.0).raise_if_cancelled
        assert _compiled_hbm._has_native_timeline_hooks(packed)
        retry = runtime.begin_transaction(owners=("hbm",), _owned_timeline=True)
        with pytest.raises(NativeFastWallBudgetError):
            core.submit(small)
        runtime.restore(retry)
        assert memo.stats["hits"] == 1
        assert tuple(array.tobytes() for array in packed.arrays) == before
        assert runtime.timeline.ready_ns["foreign"] == 14.0
        assert not any(key == "foreign" for key in retry["timeline"]["_owned_transaction"].entries["ready_ns"])


def test_corrupt_compressed_postimage_uses_sparse_rollback_without_foreign_writes():
    from heterollm_sim.data_motion import PhysicalRuntimeContext

    runtime = PhysicalRuntimeContext(configuration(channels=8, interleave_bytes=128), "hbm", capture_details=False)
    core = runtime.runtimes["hbm"].core
    runtime.timeline.reserve("unrelated-owner:lane", 111.0, 2.0)
    core.submit(AccessRequest("warm", "read", 0, 4096, 0.0))
    small = AccessRequest("small", "write", 3, 65, 1500.0)
    packed = core._hbm_numeric_state
    assert packed.request_selection(small) is not None
    before = tuple(array.tobytes() for array in packed.arrays)
    timeline_before = deepcopy({name: getattr(runtime.timeline, name) for name in
        ("ready_ns", "busy_ns", "bytes_moved", "directions", "last_intervals", "lane_available")})
    with memos.hbm_numeric_memo_experiment(min_request_bytes=0) as memo:
        seed = runtime.begin_transaction(owners=("hbm",), _owned_timeline=True)
        expected = core.submit(small)
        changed = [index for index, array in enumerate(packed.arrays) if array.tobytes() != before[index]]
        runtime.restore(seed)
        key, entry = next(iter(memo._entries.items()))
        post = list(entry[0])
        # The authentic decoder is unchanged. Damage a later compressed image
        # so at least one genuine state write precedes its decompression error.
        bad_index = next(index for index, image in enumerate(post)
                         if index > min(changed) and image[1] == memos._ZLIB_CODEC)
        frame = post[bad_index]
        post[bad_index] = frame[:3] + (bytes(len(frame[3])),)
        memo._entries[key] = (tuple(post),) + entry[1:]
        assert memos._decode_image(post[min(changed)]) != before[min(changed)]
        assert _compiled_hbm._has_native_timeline_hooks(packed)
        retry = runtime.begin_transaction(owners=("hbm",), _owned_timeline=True)
        assert "_owned_transaction" in retry["timeline"]
        with pytest.raises(memos.zlib.error):
            core.submit(small)
        # execute_hbm_numeric already restores its sparse packed checkpoint
        # before propagating the error; the owner rollback also stays valid.
        assert tuple(array.tobytes() for array in packed.arrays) == before
        runtime.restore(retry)
        assert tuple(array.tobytes() for array in packed.arrays) == before
        assert memo.stats["replay_failures"] == 1 and memo.stats["entries"] == 0
        assert memo.stats["resident_bytes"] == 0
        for name, value in timeline_before.items():
            assert getattr(runtime.timeline, name) == value
        assert not any(key.startswith("unrelated-owner:")
                       for key in retry["timeline"]["_owned_transaction"].entries["ready_ns"])
        # The failed transition is not cached; a clean retry runs natively.
        assert core.submit(small) == expected
        assert memo.stats["misses"] == 2 and memo.stats["hits"] == 0


def test_two_gib_is_a_hard_upper_limit():
    with pytest.raises(ValueError, match="2 GiB"):
        with memos.hbm_numeric_memo_experiment(max_bytes=2 * 1024**3 + 1):
            pass
