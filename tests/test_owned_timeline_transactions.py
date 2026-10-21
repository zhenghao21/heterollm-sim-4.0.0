"""Native JEDEC timeline undo is sparse, ordered, and transaction-local."""

import copy

import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest
from heterollm_sim import _compiled_hbm
from test_hbm_compiled_control import configuration
from test_hbm_resident_controller import visible
from test_token_owned_history import bits


def runtimes():
    config = configuration(channels=8, interleave_bytes=128)
    contexts = [PhysicalRuntimeContext(config, "hbm", capture_details=False) for _ in range(2)]
    cores = [context.runtimes["hbm"].core for context in contexts]
    for context, core in zip(contexts, cores):
        for index in range(200):
            context.timeline.reserve("unrelated:" + str(index), float(index), .125)
        core.submit(AccessRequest("warm", "read", 0, 4096))
    return contexts, cores


def assert_same(contexts):
    assert bits(visible(contexts[0].snapshot())) == bits(visible(contexts[1].snapshot()))


def test_owned_transaction_saves_only_resources_written_and_keeps_original_lane_view():
    contexts, cores = runtimes()
    saved = [context.begin_transaction(owners=("hbm",), _owned_timeline=index == 0)
             for index, context in enumerate(contexts)]
    journal = saved[0]["timeline"]["_owned_transaction"]
    assert not any(journal.entries.values())
    assert "_owned_transaction" not in saved[1]["timeline"]
    before = tuple(cores[0]._hbm_numeric_state.arrays[8][:, 0])
    request = AccessRequest("small", "write", 3, 65, 1500.)
    assert cores[0].submit(request) == cores[1].submit(request)
    ids = cores[0]._hbm_numeric_state.ids
    for index, key in enumerate(ids):
        lanes = saved[0]["timeline"]["lane_available"].get(key)
        if lanes is not None:
            assert lanes[0] == before[index]
    assert not any(key.startswith("unrelated:") for key in journal.entries["lane_available"])
    assert len(journal.entries["lane_available"]) < len(contexts[0].timeline.lane_available) / 4
    for context, token in zip(contexts, saved):
        context.restore(token)
    assert_same(contexts)


@pytest.mark.parametrize("inner_owned", (False, True))
def test_nested_owned_and_public_transactions_preserve_outer_rollback(inner_owned):
    contexts, cores = runtimes()
    outer = [context.begin_transaction(owners=("hbm",), _owned_timeline=index == 0)
             for index, context in enumerate(contexts)]
    prefix = AccessRequest("prefix", "write", 3, 65, 1500.)
    assert cores[0].submit(prefix) == cores[1].submit(prefix)
    inner = [context.begin_transaction(owners=("hbm",), _owned_timeline=inner_owned and index == 0)
             for index, context in enumerate(contexts)]
    suffix = AccessRequest("suffix", "read", 15 * 128 + 3, 65, 3000.)
    assert cores[0].submit(suffix) == cores[1].submit(suffix)
    for context, token in zip(contexts, inner):
        context.restore(token)
    assert cores[0].submit(suffix) == cores[1].submit(suffix)
    for context, token in zip(contexts, outer):
        context.restore(token)
    assert_same(contexts)
    assert not contexts[0].timeline._owned_transactions


def test_committed_transaction_retains_before_lanes_for_later_event_accounting():
    contexts, cores = runtimes()
    rid = cores[0]._hbm_numeric_state.ids[0]
    before = tuple(contexts[0].timeline.lane_available[rid])
    tokens = [context.begin_transaction(owners=("hbm",), _owned_timeline=index == 0)
              for index, context in enumerate(contexts)]
    request = AccessRequest("write", "write", 0, 4096, 5000.)
    assert cores[0].submit(request) == cores[1].submit(request)
    for context, token in zip(contexts, tokens):
        context.commit_transaction(token)
    assert tokens[0]["timeline"]["lane_available"][rid] == before
    assert_same(contexts)


def test_external_calendar_edits_between_transactions_remain_authoritative():
    contexts, cores = runtimes()
    rid = cores[0]._hbm_numeric_state.ids[0]
    for context in contexts:
        context.timeline.lane_available[rid][0] += 10000.
        context.timeline.ready_ns[rid] = context.timeline.lane_available[rid][0]
        context.timeline.busy_ns[rid] += .125
    tokens = [context.begin_transaction(owners=("hbm",), _owned_timeline=index == 0)
              for index, context in enumerate(contexts)]
    request = AccessRequest("edited", "read", 0, 4096, 5000.)
    assert cores[0].submit(request) == cores[1].submit(request)
    for context, token in zip(contexts, tokens):
        context.restore(token)
    assert_same(contexts)


@pytest.mark.parametrize("modified", ("submit", "calendar", "timeline", "default"))
def test_unknown_callbacks_and_public_transactions_keep_complete_timeline_copy(modified, monkeypatch):
    contexts, cores = runtimes()
    context, core = contexts[0], cores[0]
    if modified == "submit":
        original = DramCore.submit
        def submit(self, request, **kwargs):
            return original(self, request, **kwargs)
        monkeypatch.setattr(DramCore, "submit", submit)
    elif modified == "calendar":
        class Calendar(list):
            pass
        rid = core._hbm_numeric_state.ids[0]
        context.timeline.lane_available[rid] = Calendar(context.timeline.lane_available[rid])
    elif modified == "timeline":
        class Timeline(ResourceTimeline):
            pass
        context.timeline.__class__ = Timeline
    token = context.begin_transaction(owners=("hbm",), _owned_timeline=modified != "default")
    assert "_owned_transaction" not in token["timeline"]
    context.timeline.lane_available["unrelated:0"][0] += 1234.
    context.restore(token)
    assert context.timeline.lane_available["unrelated:0"][0] == .125


def test_zero_duration_lane_direction_deletion_intervals_and_metrics_restore_exactly():
    timeline = ResourceTimeline()
    timeline.reserve("before", 0., 1., direction="read")
    timeline.reserve("x", 1., 1., direction="write")
    timeline.reserve("after", 2., 1., direction="read")
    timeline.lane_directions = {"before": ["read"], "x": ["write"], "after": ["read"]}
    timeline._interval_limit = 20
    timeline._ordered_metric_touches("busy_ns", timeline.busy_ns, {"x"})
    initial = copy.deepcopy({key: value for key, value in vars(timeline).items()
                             if key != "_owned_transactions"})
    token = timeline._begin_owned_transaction()
    timeline.transfer("x", 32, 5., 32., direction="read")
    timeline.reserve("new-zero", 100., 0., direction="write")
    timeline._ordered_metric_touches("busy_ns", timeline.busy_ns, {"new-zero"})
    token.rollback()
    assert bits({key: value for key, value in vars(timeline).items()
                 if key != "_owned_transactions"}) == bits(initial)
    assert not timeline._owned_transactions


def test_nested_resource_alias_lists_restore_first_object_image_once():
    timeline = ResourceTimeline()
    shared = [1.]
    timeline.lane_available = {"x": shared, "y": shared}
    initial = copy.deepcopy(timeline.lane_available)
    outer = timeline._begin_owned_transaction()
    timeline.reserve("x", 5., 1.)
    inner = timeline._begin_owned_transaction()
    timeline.reserve("y", 7., 1.)
    inner.rollback()
    assert timeline.lane_available["x"] == [6.]
    outer.rollback()
    assert timeline.lane_available == initial
    assert timeline.lane_available["x"] is timeline.lane_available["y"]


@pytest.mark.parametrize("hook", (
    "class_execute", "instance_execute", "run_numeric", "numeric_dispatch",
    "pack", "entry", "control_entry", "publish", "consume", "capture"))
def test_replaced_numeric_callbacks_get_full_snapshot_and_restore_unrelated_resources(hook, monkeypatch):
    contexts, cores = runtimes()
    before = [context.timeline.ready_ns["unrelated:0"] for context in contexts]

    def injected(original):
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            for context in contexts:
                context.timeline.ready_ns["unrelated:0"] += 1234.
            raise RuntimeError("injected numeric callback")
        return wrapped

    with monkeypatch.context() as patch:
        if hook == "class_execute":
            patch.setattr(_compiled_hbm._HBMNumericState, "execute",
                          injected(_compiled_hbm._HBMNumericState.execute))
        elif hook == "instance_execute":
            for core in cores:
                packed = core._hbm_numeric_state
                patch.setattr(packed, "execute", injected(packed.execute))
        elif hook == "run_numeric":
            for core in cores:
                packed = core._hbm_numeric_state
                patch.setattr(packed, "run_numeric", injected(packed.run_numeric))
        elif hook == "numeric_dispatch":
            numeric = _compiled_hbm._owned_numeric_module
            patch.setattr(numeric, "_run_hbm_numeric", injected(numeric._run_hbm_numeric))
        elif hook == "capture":
            patch.setattr(_compiled_hbm._NumericResourceCapture, "finish",
                          injected(_compiled_hbm._NumericResourceCapture.finish))
        elif hook == "control_entry":
            control = _compiled_hbm._owned_control_module
            patch.setattr(control, "execute_hbm", injected(control.execute_hbm))
        else:
            name = {"pack": "_pack_hbm_numeric", "entry": "execute_hbm_numeric",
                    "publish": "publish_resource_columns", "consume": "consume_resource_columns"}[hook]
            patch.setattr(_compiled_hbm, name, injected(getattr(_compiled_hbm, name)))
        tokens = [context.begin_transaction(owners=("hbm",), _owned_timeline=index == 0)
                  for index, context in enumerate(contexts)]
        assert all("_owned_transaction" not in token["timeline"] for token in tokens)
        for core in cores:
            with pytest.raises(RuntimeError, match="injected numeric callback"):
                kwargs = {}
                if hook in {"capture", "consume"}:
                    from heterollm_sim._hbm_pack import HBMResourceAccumulator
                    kwargs["_resource_accumulator"] = HBMResourceAccumulator()
                core.submit(AccessRequest("bad", "read", 0, 4096, 5000.), **kwargs)
    for context, token in zip(contexts, tokens):
        context.restore(token)
    assert [context.timeline.ready_ns["unrelated:0"] for context in contexts] == before
    assert_same(contexts)


@pytest.mark.parametrize("kind", ("hbm", "nand", "sram"))
def test_unknown_core_callback_also_protects_sibling_owner_state(kind, monkeypatch):
    from heterollm_sim.memory_types import NandConfig, SramConfig
    from heterollm_sim.nand_core import NandCore
    from heterollm_sim.sram_core import SramCore
    config, core_type = ((configuration(channels=2), DramCore) if kind == "hbm" else
                         (NandConfig(), NandCore) if kind == "nand" else
                         (SramConfig(capacity_bytes=4096, burst_bytes=64,
                             interface_bandwidth_gb_s=64., read_bandwidth_gb_s=64.,
                             write_bandwidth_gb_s=32., read_latency_ns=2., write_latency_ns=3.,
                             ports=2, max_outstanding_requests=16), SramCore))
    context = PhysicalRuntimeContext(config, "a", capture_details=False)
    context.runtime(config, "b")
    # A built-in NAND/SRAM owner still has a narrow core snapshot even though
    # it does not qualify for the JEDEC-only sparse timeline optimization.
    control = context.begin_transaction(owners=("a",), _owned_timeline=True)
    assert set(control["runtime_state"]) == {"a"}
    context.commit_transaction(control)

    original = core_type.submit
    def unknown(core, request, **kwargs):
        if request.request_id == "bad":
            sibling = context.runtimes["b"]
            sibling.clock_ns += 1234.
            sibling.core._acceptance_ns += 99.
            raise RuntimeError("unknown owner callback")
        return original(core, request, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(core_type, "submit", unknown)
        token = context.begin_transaction(owners=("a",), _owned_timeline=True)
        assert set(token["runtime_state"]) == {"a", "b"}
        assert "_owned_transaction" not in token["timeline"]
        with pytest.raises(RuntimeError, match="unknown owner callback"):
            context.runtimes["a"].core.submit(AccessRequest("bad", "read", 0, 64))
    context.restore(token)
    assert context.runtimes["b"].clock_ns == 0.
    assert context.runtimes["b"].core._acceptance_ns == 0.


def test_unknown_cold_pack_callback_widens_owner_undo_before_first_numeric_state(monkeypatch):
    config = configuration(channels=2)
    context = PhysicalRuntimeContext(config, "a", capture_details=False)
    context.runtime(config, "b")
    assert context.runtimes["a"].core._hbm_numeric_state is None
    original = _compiled_hbm._pack_hbm_numeric
    def unknown(core, request, **kwargs):
        result = original(core, request, **kwargs)
        context.runtimes["b"].clock_ns += 1234.
        raise RuntimeError("unknown cold pack")
    with monkeypatch.context() as patch:
        patch.setattr(_compiled_hbm, "_pack_hbm_numeric", unknown)
        token = context.begin_transaction(owners=("a",), _owned_timeline=True)
        assert set(token["runtime_state"]) == {"a", "b"}
        with pytest.raises(RuntimeError, match="unknown cold pack"):
            context.runtimes["a"].core.submit(AccessRequest("bad", "read", 0, 4096))
    context.restore(token)
    assert context.runtimes["b"].clock_ns == 0.


def test_first_new_owner_unknown_submit_hook_protects_preexisting_sibling(monkeypatch):
    config = configuration(channels=2)
    context = PhysicalRuntimeContext(config, "b", capture_details=False)
    assert "a" not in context.runtimes

    def unknown(core, request, **kwargs):
        context.runtimes["b"].clock_ns += 1234.
        context.runtimes["b"].core._acceptance_ns += 99.
        raise RuntimeError("unknown newly created owner")

    with monkeypatch.context() as patch:
        patch.setattr(DramCore, "submit", unknown)
        token = context.begin_transaction(owners=("a",), _owned_timeline=True)
        assert set(token["runtime_state"]) == {"b"}
        assert not token["_owned_native_writers"]
        context.runtime(config, "a")
        with pytest.raises(RuntimeError, match="newly created owner"):
            context.runtimes["a"].core.submit(AccessRequest("bad", "read", 0, 64))
    context.restore(token)
    assert set(context.runtimes) == {"b"}
    assert context.runtimes["b"].clock_ns == 0.
    assert context.runtimes["b"].core._acceptance_ns == 0.
