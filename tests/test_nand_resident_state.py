"""Resident NAND arrays retain scalar, snapshot and rollback semantics under OCP mode."""
from copy import deepcopy
from dataclasses import replace
from random import Random

import pytest

from heterollm_sim.memory_mapping import map_nand_address
from heterollm_sim.memory_types import AccessRequest
from heterollm_sim.nand_core import NandCore
from test_generic_nand_resident import logical, pair, same, submit
from test_nand_generic_buffered_acceleration import PAGE, config
from test_token_owned_history import bits


def settings(**changes):
    """The paper configurations' shape: OCP protocol with one device-wide queue."""
    values = dict(kind="HBF", protocol_mode="ocp_hbf_0_7", host_granularity_bytes=64, channels=16,
                  targets_per_channel=1, dies_per_target=1, planes_per_lun=16)
    return config(**{**values, **changes})


def core_of(context):
    return context.runtimes["flash"].core


def read(name, address, count=64, arrival=0.):
    return AccessRequest(name, "read", address, count, arrival)


@pytest.mark.parametrize("finish_inner", ["commit", "rollback"])
@pytest.mark.parametrize("resident_at_begin", [False, True])
def test_nested_numeric_and_scalar_rollback_restores_original_native_state(finish_inner, resident_at_begin):
    contexts = pair(settings())
    if resident_at_begin:
        submit(contexts, read("seed", 0, 128 * PAGE))
    outer = [context.begin_transaction() for context in contexts]
    submit(contexts, read("scalar-first", 0))
    submit(contexts, read("bulk", 0, 128 * PAGE, 1000.))
    inner = [context.begin_transaction() for context in contexts]
    submit(contexts, AccessRequest("program", "write", 0, 2 * PAGE, 2000.))
    submit(contexts, read("scalar-inner", 0, arrival=3000.))
    for context, token in zip(contexts, inner):
        (context.commit_transaction if finish_inner == "commit" else context.restore)(token)
    same(contexts)
    submit(contexts, read("short", 0, 3 * PAGE, 4000.))
    for context, token in zip(contexts, outer):
        context.restore(token)
    same(contexts)


@pytest.mark.parametrize("finish", ["commit", "rollback"])
def test_public_snapshot_reuses_active_resident_token_after_further_execution(finish):
    context = pair(settings())[0]
    core = core_of(context)
    core.submit(read("seed", 0, 128 * PAGE))
    expected = bits(logical(core))
    outer = context.begin_transaction()
    core.submit(AccessRequest("program", "write", 0, 2 * PAGE))
    saved = context.snapshot()
    for _ in range(2):
        core.submit(read("first", 0, 2 * PAGE))
        (context.commit_transaction if finish == "commit" else context.restore)(outer)
        context.restore(saved)
        core.submit(read("second", 2 * PAGE, 2 * PAGE))
        context.restore(outer)
        assert bits(logical(core)) == expected
        context.restore(saved)


def test_exposed_mutable_state_disables_residency_and_is_protected_on_rollback():
    core = core_of(pair(settings())[0])
    core.submit(read("seed", 0, 128 * PAGE))
    cache, arrays, buffers = core._read_cache, core._array_ready, core._buffer_ready
    assert core._nand_numeric_state is None and not core._nand_numeric_enabled
    before = deepcopy((cache, arrays, buffers))
    token = core.begin_transaction()
    next(iter(cache.values()))[0].ready_ns = 1e9
    arrays[next(iter(arrays))] = 1e9
    buffers[next(iter(buffers))] = 1e9
    core.submit(read("after-edit", 0, 2 * PAGE))
    core.rollback_transaction(token)
    assert (core._read_cache, core._array_ready, core._buffer_ready) == before


def test_random_nested_transactions_match_published_page_machine():
    contexts = pair(settings())
    rng = Random(6349)
    submit(contexts, read("seed", 0, 256 * PAGE))
    for _ in range(10):
        outer = [context.begin_transaction() for context in contexts]
        inner = None
        for step in range(8):
            if step == 2:
                inner = [context.begin_transaction() for context in contexts]
            if step == 6:
                finish = "commit_transaction" if rng.randrange(2) else "restore"
                for context, token in zip(contexts, inner):
                    getattr(context, finish)(token)
            if step == 4:
                submit(contexts, read("partial", rng.randrange(128) * PAGE + 64))
                continue
            operation = "write" if rng.randrange(4) == 0 else "read"
            for _ in range(rng.randrange(2, 6)):
                submit(contexts, AccessRequest("page", operation, rng.randrange(128) * PAGE, PAGE,
                                               rng.randrange(1000) * .5))
        finish = "commit_transaction" if rng.randrange(2) else "restore"
        for context, token in zip(contexts, outer):
            getattr(context, finish)(token)
        same(contexts)


def test_scalar_handoffs_retain_the_same_resident_arrays():
    """A single-bank request must not unload the other banks between page runs."""
    contexts = pair(settings())
    submit(contexts, read("seed", 0, 256 * PAGE))
    core = core_of(contexts[0])
    packed = core._nand_numeric_state
    arrays = packed.arrays
    for request in (read("page", 128 * PAGE, PAGE), AccessRequest("program", "write", 0, PAGE),
                    read("partial", 64)):
        submit(contexts, request)
        assert core._nand_numeric_state is packed
        assert all(current is original for current, original in zip(packed.arrays, arrays))


def test_scalar_handoff_exception_retains_exact_partial_state():
    contexts = pair(settings())
    submit(contexts, read("seed", 0, 32 * PAGE))
    # An overflow raised by the original scalar reserve follows its original
    # mutation ordering. The bridge publishes the same partial state later.
    request = read("overflow", 0)
    for context in contexts:
        core = core_of(context)
        mapping = map_nand_address(core.config, request.address)
        core.timeline.busy_ns[core._host_id(mapping, request.operation)] = 10**400
        with pytest.raises(OverflowError, match="too large"):
            core.submit(request)
    same(contexts)


@pytest.mark.parametrize("method", ["_cache_slot", "_write_buffer_payload_bytes"])
def test_authored_scalar_method_replacement_receives_fully_published_state(monkeypatch, method):
    core = core_of(pair(settings())[0])
    core.submit(read("seed", 0, 128 * PAGE))
    original = getattr(NandCore, method)
    observed = []
    def authored(self, *args):
        observed.append((self._nand_numeric_state, len(self.__dict__["_read_cache"])))
        return original(self, *args)
    monkeypatch.setattr(NandCore, method, authored)
    core.submit(read("authored", 0))
    assert observed and set(observed) == {(None, 128)}


@pytest.mark.parametrize("method", ["_resident_scalar_eligible", "_submit_resident_generic"])
@pytest.mark.parametrize("scope", ["instance", "class"])
def test_authored_resident_dispatch_cannot_enter_partially_published_state(monkeypatch, method, scope):
    core = core_of(pair(settings())[0])
    core.submit(read("seed", 0, 128 * PAGE))
    def authored(*args):
        pytest.fail("unqualified resident helper was called before full publication")
    monkeypatch.setattr(core if scope == "instance" else NandCore, method, authored)
    core.submit(read("authored-resident-dispatch", 0))
    assert core._nand_numeric_state is None
    assert len(core.__dict__["_read_cache"]) == 128


def test_scalar_handoff_uses_the_same_multi_plane_bank_and_array():
    configuration = settings(channels=2, targets_per_channel=2, dies_per_target=2,
                             luns_per_die=2, planes_per_lun=2)
    contexts = pair(configuration)
    base_units = (configuration.channels * configuration.targets_per_channel
                  * configuration.dies_per_target * configuration.luns_per_die)
    submit(contexts, read("seed", 0, base_units * 2 * PAGE))
    for bank in (base_units + 1, 1, base_units * 2 - 1):
        submit(contexts, read("read-bank", bank * PAGE + 64))
        submit(contexts, AccessRequest("write-bank", "write", bank * PAGE, PAGE))


def test_scalar_cache_clock_beyond_native_range_keeps_python_integer_semantics():
    contexts = pair(settings())
    submit(contexts, read("seed", 0, 32 * PAGE))
    for context in contexts:
        core = core_of(context)
        core._nand_before_reference()
        core._cache_use = 2**63 - 1
    submit(contexts, read("large-clock", 0))
    for context in contexts:
        assert core_of(context)._cache_use == 2**63


@pytest.mark.parametrize("changed", ["array_resource_prefix", "parallel_resource_prefix",
                                     "channel_resource_prefix", "host_resource_id", "resource_alias"])
def test_scalar_handoff_publishes_before_mutable_resource_layout_changes(changed):
    contexts = pair(settings())
    for context in contexts:
        # Each context has its own metadata so the second oracle starts from
        # the same original resource identifiers.
        core = core_of(context)
        core.config = replace(core.config, metadata=dict(core.config.metadata))
    submit(contexts, read("seed", 0, 128 * PAGE))
    for context in contexts:
        core = core_of(context)
        if changed == "resource_alias":
            core.timeline.resource_aliases["nand:array:0"] = "shared:array"
        else:
            core.config.metadata[changed] = "changed:" + changed
    request = AccessRequest("changed-resources", "write", 0, PAGE)
    assert not core_of(contexts[0])._resident_scalar_eligible(request)
    submit(contexts, request)
    submit(contexts, read("after", 0, 2 * PAGE))
