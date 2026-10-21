"""Native generic NAND and SRAM share exact sparse calendar rollback."""
import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_types import AccessRequest, NandConfig, SramConfig
from heterollm_sim import _nand_acceleration
from test_token_owned_history import bits


def settings(kind):
    if kind == "sram":
        return SramConfig(capacity_bytes=2**24, burst_bytes=64,
            interface_bandwidth_gb_s=64., read_bandwidth_gb_s=64.,
            write_bandwidth_gb_s=32., read_latency_ns=2., write_latency_ns=3.,
            ports=4, max_outstanding_requests=16)
    return NandConfig(channels=2, targets_per_channel=2, dies_per_target=2,
        planes_per_lun=3, planes_independent=True, page_bytes=4096,
        read_cache_pages_per_bank=2 if kind == "nand" else 0,
        host_full_duplex=kind == "nand", host_bandwidth_gb_s=17.,
        host_read_bandwidth_gb_s=17., host_write_bandwidth_gb_s=7.,
        host_granularity_bytes=4096, internal_bandwidth_gb_s=3.1,
        page_read_ns=91.5, page_program_ns=351.1, front_ns=.1)


def pair(kind):
    contexts = [PhysicalRuntimeContext(settings(kind), "dev", capture_details=False) for _ in range(2)]
    for context in contexts:
        context.runtime(settings(kind), "other")
        for index in range(200):
            context.timeline.reserve(f"unrelated:{index}", float(index), .125)
    return contexts


def assert_same(contexts):
    saved = [context.snapshot() for context in contexts]
    for field in ("timeline", "allocator_state", "committed_owners", "hbf_staging"):
        assert bits(saved[0][field]) == bits(saved[1][field]), field
    for owner, actual in saved[0]["runtime_state"].items():
        expected = saved[1]["runtime_state"][owner]
        assert bits(actual["core"]) == bits(expected["core"]), owner
        assert bits(actual["clock_ns"]) == bits(expected["clock_ns"]), owner


@pytest.mark.parametrize("kind", ("nand", "affine", "sram"))
@pytest.mark.parametrize("finish", ("commit", "restore"))
def test_mixed_operations_keep_exact_calendars_and_physical_state(kind, finish):
    contexts = pair(kind)
    tokens = [context.begin_transaction(owners=("dev",), _owned_timeline=index == 0)
              for index, context in enumerate(contexts)]
    assert "_owned_transaction" in tokens[0]["timeline"]
    journal = tokens[0]["timeline"]["_owned_transaction"]
    requests = (
        AccessRequest("bulk", "read", 0, 256 * 4096, .125),
        AccessRequest("write", "write", 7 * 4096 + 33, 4199, 111.25),
        AccessRequest("cached", "read", 4096, 8192, 700.001),
    )
    for request in requests:
        results = [context.runtimes["dev"].core.submit(request) for context in contexts]
        assert bits(results[0]) == bits(results[1])
    assert not any(key.startswith("unrelated:") for key in journal.entries["ready_ns"])
    for context, token in zip(contexts, tokens):
        getattr(context, "commit_transaction" if finish == "commit" else "restore")(token)
    assert_same(contexts)


@pytest.mark.parametrize("finish", ("commit", "restore"))
@pytest.mark.parametrize("multilane", (False, True))
def test_actual_pd_nand_selected_pack_matches_full_pack_and_proof_lifetime(finish, multilane):
    from test_nand_generic_buffered_acceleration import config, PAGE
    settings = config(channels=112, targets_per_channel=1, dies_per_target=1,
        planes_per_lun=25, planes_independent=True)
    contexts = [PhysicalRuntimeContext(settings, "dev", capture_details=False) for _ in range(2)]
    for context in contexts:
        core = context.runtimes["dev"].core
        core.submit(AccessRequest("warm", "write", 0, 4096*PAGE, .125))
        if multilane:
            host = core.config.metadata["host_resource_id"] + ":write"
            context.timeline.lane_available[host] = [0., 123.25]
    saved = [context.begin_transaction(owners=("dev",), _owned_timeline=index == 0)
             for index, context in enumerate(contexts)]
    assert saved[0]["runtime_state"]["dev"]["nand_transaction"].native_timeline_qualified
    assert not saved[1]["runtime_state"]["dev"]["nand_transaction"].native_timeline_qualified
    for index, (operation, address) in enumerate((("write", 4096*PAGE), ("read", 4096*PAGE),
            ("write", 7201*PAGE), ("read", 7201*PAGE))):
        values = [context.runtimes["dev"].core.submit(AccessRequest(str(index), operation,
                  address, 256*PAGE, 100000.125 + index)) for context in contexts]
        assert bits(values[0]) == bits(values[1])
    for context, checkpoint in zip(contexts, saved):
        getattr(context, "commit_transaction" if finish == "commit" else "restore")(checkpoint)
        assert not context.runtimes["dev"].core._nand_transactions
    assert_same(contexts)
    public = contexts[0].runtimes["dev"].core.begin_transaction()
    assert not public.native_timeline_qualified
    contexts[0].runtimes["dev"].core.commit_transaction(public)


@pytest.mark.parametrize("kind", ("nand", "sram"))
@pytest.mark.parametrize("inner_owned", (False, True))
def test_nested_rollback_retains_earlier_owner_checkpoint(kind, inner_owned):
    contexts = pair(kind)
    outer = [context.begin_transaction(owners=("dev",), _owned_timeline=index == 0)
             for index, context in enumerate(contexts)]
    for context in contexts:
        context.runtimes["dev"].core.submit(AccessRequest("prefix", "write", 0, 4096, .125))
    inner = [context.begin_transaction(owners=("dev",), _owned_timeline=inner_owned and index == 0)
             for index, context in enumerate(contexts)]
    for context in contexts:
        context.runtimes["dev"].core.submit(AccessRequest("suffix", "read", 0, 256 * 4096, 111.25))
    for context, token in zip(contexts, inner):
        context.restore(token)
    for context, token in zip(contexts, outer):
        context.restore(token)
    assert_same(contexts)
    assert not contexts[0].timeline._owned_transactions


@pytest.mark.parametrize("kind", ("nand", "sram"))
def test_plain_aliases_and_shared_lane_lists_restore_original_objects(kind):
    contexts = pair(kind)
    for context in contexts:
        core = context.runtimes["dev"].core
        if kind == "nand":
            original = _nand_acceleration.owned_resource_footprint(core)[-1]
        else:
            original = core.config.metadata["port_resource_prefix"] + ":0"
        context.timeline.resource_aliases[original] = "shared-wire"
        shared = [123.25]
        context.timeline.lane_available["shared-wire"] = shared
        context.timeline.lane_available["another-alias"] = shared
    tokens = [context.begin_transaction(owners=("dev",), _owned_timeline=index == 0)
              for index, context in enumerate(contexts)]
    assert "_owned_transaction" in tokens[0]["timeline"]
    for context in contexts:
        context.runtimes["dev"].core.submit(AccessRequest("alias", "read", 0, 256 * 4096, .125))
    for context, token in zip(contexts, tokens):
        context.restore(token)
    assert_same(contexts)
    assert contexts[0].timeline.lane_available["shared-wire"] is contexts[0].timeline.lane_available["another-alias"]


@pytest.mark.parametrize("kind", ("nand", "sram"))
def test_unknown_calendar_callback_uses_full_timeline_and_owner_undo(kind):
    contexts = pair(kind)
    context = contexts[0]
    core = context.runtimes["dev"].core
    if kind == "nand":
        rid = _nand_acceleration.owned_resource_footprint(core)[0]
    else:
        rid = core.config.metadata["port_resource_prefix"] + ":0"
    class Calendar(list):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            context.runtimes["other"].clock_ns += 1234.
            context.timeline.ready_ns["unrelated:0"] += 99.
            raise RuntimeError("authored calendar")
    context.timeline.lane_available[rid] = Calendar([0.])
    token = context.begin_transaction(owners=("dev",), _owned_timeline=True)
    assert "_owned_transaction" not in token["timeline"]
    assert set(token["runtime_state"]) == {"dev", "other"}
    with pytest.raises(RuntimeError, match="authored calendar"):
        core.submit(AccessRequest("bad", "read", 0, 256 * 4096))
    context.restore(token)
    assert context.runtimes["other"].clock_ns == 0.
    assert context.timeline.ready_ns["unrelated:0"] == .125


def test_replaced_numeric_publisher_has_full_rollback_for_foreign_owner(monkeypatch):
    context = pair("nand")[0]
    pack = _nand_acceleration._owned_pack_module
    original = pack.publish_timeline
    def unknown(*args, **kwargs):
        original(*args, **kwargs)
        context.timeline.ready_ns["unrelated:0"] += 99.
        context.runtimes["other"].clock_ns += 1234.
        raise RuntimeError("unknown publication")
    with monkeypatch.context() as patch:
        patch.setattr(pack, "publish_timeline", unknown)
        token = context.begin_transaction(owners=("dev",), _owned_timeline=True)
        assert "_owned_transaction" not in token["timeline"]
        assert set(token["runtime_state"]) == {"dev", "other"}
        with pytest.raises(RuntimeError, match="unknown publication"):
            context.runtimes["dev"].core.submit(AccessRequest("bad", "read", 0, 256 * 4096))
    context.restore(token)
    assert context.runtimes["other"].clock_ns == 0.
    assert context.timeline.ready_ns["unrelated:0"] == .125


def test_unknown_alias_target_is_rejected_before_qualification_hashes_it():
    context = pair("nand")[0]
    core = context.runtimes["dev"].core
    rid = _nand_acceleration.owned_resource_footprint(core)[0]
    class ForeignWire(str):
        def __hash__(self):
            context.runtimes["other"].clock_ns += 99.
            raise RuntimeError("authored alias target")
    context.timeline.resource_aliases[rid] = ForeignWire("wire")
    token = context.begin_transaction(owners=("dev",), _owned_timeline=True)
    assert set(token["runtime_state"]) == {"dev", "other"}
    assert "_owned_transaction" not in token["timeline"]
    with pytest.raises(RuntimeError, match="authored alias target"):
        core.submit(AccessRequest("bad", "read", 0, 256 * 4096))
    context.restore(token)
    assert context.runtimes["other"].clock_ns == 0.


def test_native_failure_finishes_inner_metric_journal_before_outer_restore():
    config = NandConfig(channels=1, targets_per_channel=1, dies_per_target=1,
        page_read_ns=1e308, page_program_ns=1e308)
    context = PhysicalRuntimeContext(config, "dev", capture_details=False)
    token = context.begin_transaction(owners=("dev",), _owned_timeline=True)
    assert len(context.timeline._owned_transactions) == 1
    with pytest.raises(ValueError, match="finite timing range"):
        context.runtimes["dev"].core.submit(AccessRequest("overflow", "read", 0, 4096, 1e308))
    assert len(context.timeline._owned_transactions) == 1
    context.restore(token)
    assert not context.timeline._owned_transactions
    assert not context.timeline.ready_ns


def test_private_core_metrics_keep_original_values_with_independent_nested_journal():
    context = pair("sram")[0]
    timeline = context.timeline
    public = timeline._core_metrics_snapshot()
    assert type(public["busy_ns"]) is dict
    outer = context.begin_transaction(owners=("dev",), _owned_timeline=True)
    before = timeline._core_metrics_snapshot()
    assert type(before["busy_ns"]) is not dict
    timeline.reserve("unrelated:0", 999., .25)
    assert before["busy_ns"]["unrelated:0"] == .125
    metrics = timeline._core_metrics_delta(before)
    assert metrics["resource_busy_ns"] == {"unrelated:0": .25}
    assert len(timeline._owned_transactions) == 1
    context.restore(outer)
    assert timeline.busy_ns["unrelated:0"] == .125


def test_generic_frontier_undo_saves_only_changed_keys_and_matches_exposed_full_copy():
    contexts = pair("nand")
    for context in contexts:
        # This case exercises the object-frontier journal independently of
        # the private numeric controller, whose arrays have their own undo.
        context.runtimes["dev"].core._nand_numeric_enabled = False
        context.runtimes["dev"].core.submit(AccessRequest("warm", "read", 0, 256 * 4096))
    cores = [context.runtimes["dev"].core for context in contexts]
    cores[1]._nand_state_exposed = True
    tokens = [core.begin_transaction() for core in cores]
    assert tokens[0].array_frontiers is None
    assert not any(tokens[0].frontier_undo)
    assert tokens[1].array_frontiers is not None
    request = AccessRequest("one-page", "write", 0, 4096, 10000.)
    assert bits(cores[0].submit(request)) == bits(cores[1].submit(request))
    assert [len(undo) for undo in tokens[0].frontier_undo] == [1, 1]
    for core, token in zip(cores, tokens):
        core.rollback_transaction(token)
    for key in ("_array_ready", "_buffer_ready", "_read_cache"):
        assert bits(cores[0].__dict__[key]) == bits(cores[1].__dict__[key])


def test_public_frontier_alias_materializes_earlier_lazy_undo_before_external_edit():
    context = pair("nand")[0]
    core = context.runtimes["dev"].core
    core._nand_numeric_enabled = False
    core.submit(AccessRequest("warm", "read", 0, 256 * 4096))
    before = dict(core.__dict__["_array_ready"])
    token = core.begin_transaction()
    core.submit(AccessRequest("prefix", "write", 0, 4096, 10000.))
    alias = core._array_ready
    assert token.array_frontiers is not None
    alias["external"] = 1234.
    core.rollback_transaction(token)
    assert core.__dict__["_array_ready"] == before
    assert alias["external"] == 1234.


@pytest.mark.parametrize("runtime_transaction", (False, True))
def test_unknown_submit_uses_full_frontier_copy_for_direct_writes(monkeypatch, runtime_transaction):
    context = pair("nand")[0]
    core = context.runtimes["dev"].core
    original = type(core).submit
    def unknown(current, request):
        if request.request_id == "bad":
            current.__dict__["_array_ready"]["callback-array"] = 99.
            current.__dict__["_buffer_ready"]["callback-buffer"] = 101.
            raise RuntimeError("unknown frontier writer")
        return original(current, request)
    with monkeypatch.context() as patch:
        patch.setattr(type(core), "submit", unknown)
        if runtime_transaction:
            token = context.begin_transaction(owners=("dev",), _owned_timeline=True)
            frontier = token["runtime_state"]["dev"]["nand_transaction"]
        else:
            token = frontier = core.begin_transaction()
        assert frontier.array_frontiers is not None
        with pytest.raises(RuntimeError, match="unknown frontier writer"):
            core.submit(AccessRequest("bad", "read", 0, 4096))
    if runtime_transaction:
        context.restore(token)
    else:
        core.rollback_transaction(token)
    assert "callback-array" not in core.__dict__["_array_ready"]
    assert "callback-buffer" not in core.__dict__["_buffer_ready"]
