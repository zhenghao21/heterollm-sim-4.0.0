"""HOST numeric execution retains public aliases and transaction semantics."""
from copy import deepcopy
from dataclasses import replace

import pytest

from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_dram_core_semantics import config


@pytest.mark.parametrize("size", (65, 4096, 1048576))
def test_host_owned_timeline_matches_full_undo_with_foreign_gpu_calendars(size):
    from heterollm_sim.data_motion import PhysicalRuntimeContext
    from test_token_owned_history import bits
    contexts = [PhysicalRuntimeContext(config(channels=2, rows_per_bank=32768),
                "host", capture_details=False) for _ in range(2)]
    for context in contexts:
        for index in range(1000):
            context.timeline.reserve(f"foreign-gpu:{index}", index+.125, .375)
        context.runtimes["host"].core.submit(AccessRequest("warm", "write", 0, 4096))
    tokens = [context.begin_transaction(owners=("host",), _owned_timeline=index == 0)
              for index, context in enumerate(contexts)]
    journal = tokens[0]["timeline"]["_owned_transaction"]
    assert not any(key.startswith("foreign-gpu:") for key in journal.entries["lane_available"])
    for address, operation in ((3, "read"), (131075, "write")):
        values = [context.runtimes["host"].core.submit(AccessRequest("next", operation,
                  address, size, 100000.)) for context in contexts]
        assert values[0] == values[1]
    for context, token in zip(contexts, tokens):
        context.restore(token)
    def state(context):
        snapshot = context.snapshot()
        return (snapshot["runtime_state"]["host"]["core"], snapshot["timeline"],
                snapshot["allocator_state"], snapshot["committed_owners"])
    assert bits(state(contexts[0])) == bits(state(contexts[1]))


def test_host_unknown_numeric_callback_protects_foreign_owner_and_timeline(monkeypatch):
    from heterollm_sim.data_motion import PhysicalRuntimeContext
    from heterollm_sim import _compiled_dram
    context = PhysicalRuntimeContext(config(rows_per_bank=32768), "host", capture_details=False)
    context.runtime(config(rows_per_bank=32768), "other")
    context.timeline.reserve("foreign-gpu", 0., 1.)
    def callback(*args, **kwargs):
        context.runtimes["other"].clock_ns = 1234.
        context.timeline.ready_ns["foreign-gpu"] = 9999.
        raise ValueError("unknown numeric callback")
    monkeypatch.setattr(_compiled_dram, "execute_compiled", callback)
    token = context.begin_transaction(owners=("host",), _owned_timeline=True)
    assert "_owned_transaction" not in token["timeline"]
    assert set(token["runtime_state"]) == {"host", "other"}
    with pytest.raises(ValueError, match="unknown numeric callback"):
        context.runtimes["host"].core.submit(AccessRequest("x", "read", 0, 4096))
    context.restore(token)
    assert context.runtimes["other"].clock_ns == 0.
    assert context.timeline.ready_ns["foreign-gpu"] == 1.


@pytest.mark.parametrize("cached_native_type", (False, True))
def test_unknown_cached_host_workspace_cannot_escape_foreign_owner_rollback(monkeypatch, cached_native_type):
    from heterollm_sim.data_motion import PhysicalRuntimeContext
    from heterollm_sim import _compiled_dram
    import numpy as np

    context = PhysicalRuntimeContext(config(rows_per_bank=32768), "host", capture_details=False)
    context.runtime(config(rows_per_bank=32768), "other")
    context.timeline.reserve("foreign-gpu", 0., 1.)
    core = context.runtimes["host"].core
    core.submit(AccessRequest("warm", "read", 0, 4096))

    def mutate():
        context.runtimes["other"].clock_ns = 1234.
        context.timeline.ready_ns["foreign-gpu"] = 9999.
        raise ValueError("unknown workspace callback")

    if cached_native_type:
        class CallbackArray(np.ndarray):
            def fill(self, value):
                mutate()
        workspace = _compiled_dram._WORKSPACES[core]
        arrays = list(workspace.arrays)
        arrays[3] = arrays[3].view(CallbackArray)
        workspace.arrays = tuple(arrays)
    else:
        class ForeignWorkspace:
            @property
            def shape(self):
                mutate()
        monkeypatch.setitem(_compiled_dram._WORKSPACES, core, ForeignWorkspace())
    token = context.begin_transaction(owners=("host",), _owned_timeline=True)
    assert "_owned_transaction" not in token["timeline"]
    assert set(token["runtime_state"]) == {"host", "other"}
    with pytest.raises(ValueError, match="unknown workspace callback"):
        core.submit(AccessRequest("next", "read", 0, 4096))
    context.restore(token)
    assert context.runtimes["other"].clock_ns == 0.
    assert context.timeline.ready_ns["foreign-gpu"] == 1.


def test_host_metadata_protocol_is_not_invoked_while_qualifying_sparse_undo():
    from heterollm_sim.data_motion import PhysicalRuntimeContext

    context = PhysicalRuntimeContext(config(rows_per_bank=32768), "host", capture_details=False)
    context.runtime(config(rows_per_bank=32768), "other")
    context.timeline.reserve("foreign-gpu", 0., 1.)
    core = context.runtimes["host"].core
    calls = []
    class Metadata(dict):
        def get(self, *args):
            calls.append(args)
            context.runtimes["other"].clock_ns = 1234.
            context.timeline.ready_ns["foreign-gpu"] = 9999.
            raise ValueError("unknown metadata callback")
    core.config = replace(core.config, metadata=Metadata(core.config.metadata))
    token = context.begin_transaction(owners=("host",), _owned_timeline=True)
    assert not calls and "_owned_transaction" not in token["timeline"]
    with pytest.raises(ValueError, match="unknown metadata callback"):
        core.submit(AccessRequest("next", "read", 0, 4096))
    context.restore(token)
    assert context.runtimes["other"].clock_ns == 0.
    assert context.timeline.ready_ns["foreign-gpu"] == 1.


@pytest.mark.parametrize("kind", ("config_method", "scalar_protocol"))
def test_host_authored_config_callback_keeps_complete_foreign_undo(monkeypatch, kind):
    from heterollm_sim.data_motion import PhysicalRuntimeContext
    from heterollm_sim.memory_types import DramConfig

    context = PhysicalRuntimeContext(config(rows_per_bank=32768), "host", capture_details=False)
    context.runtime(config(rows_per_bank=32768), "other")
    context.timeline.reserve("foreign-gpu", 0., 1.)
    core = context.runtimes["host"].core
    calls = []
    def mutate(*args, **kwargs):
        calls.append(1)
        context.runtimes["other"].clock_ns = 1234.
        context.timeline.ready_ns["foreign-gpu"] = 9999.
        raise ValueError("unknown config callback")
    if kind == "config_method":
        monkeypatch.setattr(DramConfig, "directional_lane_bandwidth_gb_s", mutate)
    else:
        class Channels(int):
            __mul__ = mutate
        object.__setattr__(core.config, "channels", Channels(core.config.channels))
    token = context.begin_transaction(owners=("host",), _owned_timeline=True)
    assert not calls and "_owned_transaction" not in token["timeline"]
    with pytest.raises(ValueError, match="unknown config callback"):
        core.submit(AccessRequest("next", "read", 0, 4096))
    context.restore(token)
    assert context.runtimes["other"].clock_ns == 0.
    assert context.timeline.ready_ns["foreign-gpu"] == 1.


def test_public_host_bank_callback_keeps_foreign_owner_complete_undo():
    from heterollm_sim.data_motion import PhysicalRuntimeContext

    context = PhysicalRuntimeContext(config(rows_per_bank=32768), "host", capture_details=False)
    context.runtime(config(rows_per_bank=32768), "other")
    context.timeline.reserve("foreign-gpu", 0., 1.)
    core = context.runtimes["host"].core
    core.submit(AccessRequest("warm", "read", 0, 4096))
    calls = []
    class Bank:
        open_row = 0
        activation_ready_ns = 0.
        @property
        def ready_ns(self):
            calls.append(1)
            context.runtimes["other"].clock_ns = 1234.
            context.timeline.ready_ns["foreign-gpu"] = 9999.
            raise ValueError("unknown bank callback")
    public = core._banks
    public[next(iter(public))] = Bank()
    token = context.begin_transaction(owners=("host",), _owned_timeline=True)
    assert not calls and "_owned_transaction" not in token["timeline"]
    with pytest.raises(ValueError, match="unknown bank callback"):
        core.submit(AccessRequest("next", "read", 0, 4096))
    context.restore(token)
    assert context.runtimes["other"].clock_ns == 0.
    assert context.timeline.ready_ns["foreign-gpu"] == 1.


def test_public_native_host_bank_can_alias_a_sibling_owner_and_keeps_complete_undo():
    from heterollm_sim.data_motion import PhysicalRuntimeContext

    context = PhysicalRuntimeContext(config(rows_per_bank=32768), "host", capture_details=False)
    context.runtime(config(rows_per_bank=32768), "other")
    host, other = (context.runtimes[name].core for name in ("host", "other"))
    for core in (host, other):
        core.submit(AccessRequest("warm", "read", 0, 4096))
    banks = host._banks
    identity = next(iter(banks))
    banks[identity] = next(iter(other._banks.values()))
    before = deepcopy(other.__dict__["_banks"])
    token = context.begin_transaction(owners=("host",), _owned_timeline=True)
    assert "_owned_transaction" not in token["timeline"]
    assert set(token["runtime_state"]) == {"host", "other"}
    host.submit(AccessRequest("next", "write", 262144, 4096, arrival_ns=100000.))
    assert other.__dict__["_banks"] != before
    context.restore(token)
    assert other.__dict__["_banks"] == before


def test_native_host_bank_deepcopy_slotname_cache_does_not_disable_owned_undo():
    from heterollm_sim.data_motion import PhysicalRuntimeContext

    context = PhysicalRuntimeContext(config(rows_per_bank=32768), "host", capture_details=False)
    core = context.runtimes["host"].core
    core.submit(AccessRequest("warm", "read", 0, 4096))
    # copy._slotnames adds this harmless class cache on the first ordinary
    # deepcopy, which occurs in native layer-window source copies too.
    deepcopy(next(iter(core.__dict__["_banks"].values())))
    token = context.begin_transaction(owners=("host",), _owned_timeline=True)
    assert "_owned_transaction" in token["timeline"]
    context.restore(token)


def populated():
    core = DramCore(config(channels=2, bank_groups_per_rank=4, banks_per_group=4,
        rows_per_bank=128, row_bytes=8192), capture_details=False)
    core.submit(AccessRequest("initialize", "write", 0, 8192))
    return core


def test_host_bank_journal_only_copies_modified_banks_and_nested_rollback():
    core = populated()
    before = deepcopy(core.__dict__["_banks"])
    outer = core.begin_transaction()
    assert outer.full is None
    core.submit(AccessRequest("small", "read", 0, 256))
    assert len(outer.positions) == 4 < len(before)
    after_outer = deepcopy(core.__dict__["_banks"])
    inner = core.begin_transaction()
    core.submit(AccessRequest("different-row", "write", 262144 + 512, 255))
    core.rollback_transaction(inner)
    assert core.__dict__["_banks"] == after_outer
    inner = core.begin_transaction()
    core.submit(AccessRequest("tiny", "read", 3, 16))  # scalar burst path
    core.commit_transaction(inner)
    core.rollback_transaction(outer)
    assert core.__dict__["_banks"] == before


@pytest.mark.parametrize("expose_before_begin", [False, True])
def test_host_public_bank_reference_keeps_full_undo_of_external_mutation(expose_before_begin):
    core = populated()
    before = deepcopy(core.__dict__["_banks"])
    retained = core._banks if expose_before_begin else None
    token = core.begin_transaction()
    assert (token.full is not None) == expose_before_begin
    core.submit(AccessRequest("small", "write", 0, 256))
    retained = core._banks if retained is None else retained
    assert token.full is not None
    bank = next(iter(retained.values()))
    bank.ready_ns = 1e20
    bank.extension = {"external": [1, 2, 3]}
    core.rollback_transaction(token)
    assert core.__dict__["_banks"] == before
    assert all(not hasattr(bank, "extension") for bank in core.__dict__["_banks"].values())


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_short_host_pack_matches_scalar_across_rows_dies_and_stacks(independent, shared):
    configured = config(stacks=2, dies_per_stack=2, channels=3,
        bank_groups_per_rank=2, banks_per_group=3, rows_per_bank=2,
        row_bytes=256, interleave_bytes=128,
        stack_interface_mode="independent" if independent else "shared",
        metadata={"shared_command_resource": "shared-host-command"} if shared else {})
    old = DramCore(configured, capture_details=False)
    old._can_accelerate = lambda: False
    new = DramCore(configured, capture_details=False)
    capacity = configured.effective_capacity_bytes
    for index, address in enumerate((3, 8191, capacity // 2 - 129, capacity - 3001, 191)):
        request = AccessRequest(str(index), "read" if index % 2 else "write",
                                address, 2047, index * 3.7)
        expected, actual = old.submit(request), new.submit(request)
        assert replace(actual, counters={}) == replace(expected, counters={})
        assert {key: actual.counters[key] for key in expected.counters} == expected.counters
        assert old._banks == new._banks
        for name in ("ready_ns", "busy_ns", "bytes_moved", "last_intervals", "directions"):
            assert getattr(old.timeline, name) == getattr(new.timeline, name)


def test_host_acceleration_rechecks_mutable_metadata_and_resource_aliases():
    core = populated()
    assert core._can_accelerate()
    core.config.metadata["command_resource_prefix"] = "alternate-host-command"
    core.timeline.resource_aliases["alternate-host-command:0"] = "foreign-command"
    assert not core._can_accelerate()
    core.timeline.resource_aliases.pop("alternate-host-command:0")
    result = core.submit(AccessRequest("new-prefix", "read", 0, 256))
    assert result.counters["compiled_float64"]
    assert "alternate-host-command:0" in result.counters["resource_busy_ns"]
    data_slot = core.timeline.lane_available["dram:data:0"]
    core.timeline.lane_available["alternate-host-command:1"] = data_slot
    assert not core._can_accelerate()


@pytest.mark.parametrize("size", [256, 4096])
def test_public_shared_bank_state_preserves_serial_updates(size):
    from heterollm_sim.dram_core import _BankState

    configured = config(channels=1, bank_groups_per_rank=1,
                        banks_per_group=2, rows_per_bank=128)
    old = DramCore(configured, capture_details=False)
    old._can_accelerate = lambda: False
    new = DramCore(configured, capture_details=False)
    first, second = "dram:bank:0:0:0:0:0:0", "dram:bank:0:0:0:0:0:1"
    references = []
    for core in (old, new):
        retained = core._banks
        retained[first], retained[second] = _BankState(), _BankState()
        if core is new:
            assert core._can_accelerate()
        # The alias can be introduced through an already retained dictionary.
        retained[second] = retained[first]
        references.append(retained[first])
    assert not new._can_accelerate()
    for request in (AccessRequest("read", "read", 0, size),
                    AccessRequest("conflict", "write", 8192, size)):
        expected, actual = old.submit(request), new.submit(request)
        assert actual == expected
        assert vars(old.timeline) == vars(new.timeline)
        assert old.__dict__["_banks"] == new.__dict__["_banks"]
        for core, shared in zip((old, new), references):
            assert core.__dict__["_banks"][first] is shared
            assert core.__dict__["_banks"][second] is shared
        assert references[0] == references[1]
        if request.request_id == "read" and size == 256:
            assert actual.counters["row_hits"] == 3
            assert actual.counters["row_misses"] == 1
            assert actual.counters["resource_busy_ns"][first] == configured.open_ns


def test_reused_workspace_observes_mutable_banks_calendars_and_restored_state():
    from heterollm_sim._compiled_dram import _WORKSPACES

    new = populated()
    old = deepcopy(new)
    old._can_accelerate = lambda: False
    workspace = _WORKSPACES[new]
    for core in (old, new):
        bank = next(iter(core._banks.values()))
        bank.open_row = None
        bank.ready_ns = 90000.
        bank.activation_ready_ns = 85000.
        core.timeline.lane_available["dram:data:0"][0] = 120000.
        core.timeline.directions["dram:data:0"] = "write"
    saved = [deepcopy(core.__dict__) for core in (old, new)]
    requests = (AccessRequest("short", "read", 1, 255),
                AccessRequest("other-row", "write", 262144 + 512, 4097),
                AccessRequest("again", "read", 65, 511))
    for repeat in range(2):
        if repeat:
            # This is also how a public runtime restore repopulates core
            # dictionaries. The cached scratch must never act as live state.
            for core, state in zip((old, new), saved):
                core.__dict__.clear()
                core.__dict__.update(deepcopy(state))
        for request in requests:
            expected, actual = old.submit(request), new.submit(request)
            assert replace(actual, counters={}) == replace(expected, counters={})
            assert {key: actual.counters[key] for key in expected.counters} == expected.counters
            assert old._banks == new._banks
            for name in ("ready_ns", "busy_ns", "bytes_moved", "last_intervals", "directions"):
                assert getattr(old.timeline, name) == getattr(new.timeline, name)
    assert _WORKSPACES[new] is workspace


def test_numeric_exception_does_not_poison_reusable_workspace(monkeypatch):
    import numpy as np
    from heterollm_sim import _dram_numeric
    from heterollm_sim._compiled_dram import _WORKSPACES, execute_compiled

    core = populated()
    before = deepcopy(core.__dict__["_banks"])
    original = _dram_numeric.run_numeric_short
    def fail(*args):
        _WORKSPACES[core].floats.fill(float("nan"))
        raise RuntimeError("injected numeric failure")
    monkeypatch.setattr(_dram_numeric, "run_numeric_short", fail)
    with pytest.raises(RuntimeError, match="injected numeric failure"):
        execute_compiled(core, AccessRequest("failed", "read", 0, 256))
    assert core.__dict__["_banks"] == before
    assert np.isfinite(_WORKSPACES[core].floats).all()
    monkeypatch.setattr(_dram_numeric, "run_numeric_short", original)
    assert execute_compiled(core, AccessRequest("next", "read", 1024, 256)) is not None


@pytest.mark.parametrize("foreign_count", [0, 200])
@pytest.mark.parametrize("shared", [False, True])
def test_internal_host_submit_preserves_all_accounting_and_state(foreign_count, shared):
    from heterollm_sim._io_execution import HostTaskScope

    configured = config(channels=2, bank_groups_per_rank=2, banks_per_group=4,
        rows_per_bank=128, row_bytes=8192, max_outstanding_requests=2,
        metadata={"shared_command_resource": "host-command"} if shared else {})
    old = DramCore(configured, capture_details=False)
    new = DramCore(configured, capture_details=False)
    for core in (old, new):
        for index in range(foreign_count):
            core.timeline.transfer(f"foreign:{index}", 11, 3.7, 9.3)
    scope = HostTaskScope(new)
    for index in range(40):
        request = AccessRequest(str(index), "write" if index % 3 else "read",
            index % 4 * 262144 + 31, (255, 1024, 4096, 8191)[index % 4], index * .7)
        expected, actual = old.submit(request), scope.submit(request)
        assert actual is not None and actual == expected
        for field in ("resource_busy_ns", "resource_bytes", "resource_last_intervals"):
            assert list(actual.counters[field].items()) == list(expected.counters[field].items())
        scope.publish()
        assert new.__dict__["_banks"] == old.__dict__["_banks"]
        for field in ("_inflight", "_acceptance_ns", "_scheduler_streak", "_scheduler_direction"):
            assert getattr(new, field) == getattr(old, field)
        for field, value in vars(old.timeline).items():
            if field != "_metric_key_orders":  # pure reducer ordering cache
                assert getattr(new.timeline, field) == value
        if index == 20:
            # Both a retained mutable calendar and same-size resource-map
            # reordering remain visible to the internal submit path.
            for core in (old, new):
                core.timeline.lane_available["dram:data:0"][0] += 1e6
                rid = next(iter(core.timeline.busy_ns))
                core.timeline.busy_ns[rid] = core.timeline.busy_ns.pop(rid)


def test_internal_host_submit_declines_custom_or_tiny_paths_and_rolls_back(monkeypatch):
    from heterollm_sim._io_execution import HostTaskScope
    from heterollm_sim import _dram_numeric
    core = populated()
    request = AccessRequest("ordinary", "read", 0, 256)
    scope = HostTaskScope(core)
    assert scope.submit(AccessRequest("tiny", "read", 0, 32)) is None
    core.submit = lambda request: "custom"
    assert HostTaskScope(core).submit(request) is None
    del core.submit
    saved_banks = deepcopy(core.__dict__["_banks"])
    saved_timeline = deepcopy(vars(core.timeline))
    token = core.begin_transaction()
    assert scope.submit(request) is not None
    scope.publish()
    core.rollback_transaction(token)
    assert core.__dict__["_banks"] == saved_banks
    # The runtime owns the shared timeline rollback separately.
    vars(core.timeline).clear()
    vars(core.timeline).update(deepcopy(saved_timeline))
    before = deepcopy(core.__dict__["_banks"])
    def fail(*args):
        raise RuntimeError("injected failure")
    monkeypatch.setattr(_dram_numeric, "run_numeric_short", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        scope.submit(request)
    assert core.__dict__["_banks"] == before
    assert vars(core.timeline) == saved_timeline


def test_host_task_scope_reuses_numeric_banks_without_changing_reports_or_final_state():
    from heterollm_sim._io_execution import HostTaskScope
    old, new = populated(), populated()
    scope = HostTaskScope(new)
    requests = [AccessRequest(str(index), "read" if index % 3 else "write",
        (index % 4) * 262144 + 65 * index, (256, 1023, 8191)[index % 3], index * .125)
        for index in range(30)]
    expected = [old.submit(request) for request in requests]
    actual = [scope.submit(request) for request in requests]
    assert actual == expected
    assert scope.workspace is not None
    scope.publish()
    assert new.__dict__["_banks"] == old.__dict__["_banks"]
    assert list(new.__dict__["_banks"]) == list(old.__dict__["_banks"])
    for field, value in vars(old.timeline).items():
        if field != "_metric_key_orders":
            assert getattr(new.timeline, field) == value
            if isinstance(value, dict):
                assert list(getattr(new.timeline, field)) == list(value)


def test_host_task_scope_flushes_each_shared_core_handoff():
    from heterollm_sim._io_execution import _HostTask
    from heterollm_sim.memory_transfer import ResourceTimeline
    old_timeline, new_timeline = ResourceTimeline(), ResourceTimeline()
    old = [DramCore(config(), old_timeline, capture_details=False) for _ in range(2)]
    new = [DramCore(config(), new_timeline, capture_details=False) for _ in range(2)]
    scope = _HostTask(None)
    for index in (0, 1, 0, 1, 1, 0):
        request = AccessRequest(str(index), "read" if index else "write", index * 256, 256, 0.)
        assert scope.submit(new[index], request) == old[index].submit(request)
    scope.flush()
    for first, second in zip(old, new):
        assert first.__dict__["_banks"] == second.__dict__["_banks"]
    for field, value in vars(old_timeline).items():
        if field != "_metric_key_orders":
            assert getattr(new_timeline, field) == value
