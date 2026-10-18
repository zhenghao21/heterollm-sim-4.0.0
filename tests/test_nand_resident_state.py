"""Resident strict NAND arrays retain scalar, snapshot and rollback semantics."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_types import AccessRequest
from heterollm_sim.physical_presets import resolve_physical_config


def contexts():
    config = resolve_physical_config("hbf_ocp_0_7")
    result = [PhysicalRuntimeContext(config=config, physical_owner="hbf", capture_details=False)
              for _ in range(2)]
    result[0].runtimes["hbf"].core._nand_numeric_enabled = False
    return result


def published(runtime):
    core = runtime.runtimes["hbf"].core
    core._nand_before_reference()
    values = {key: value for key, value in vars(core).items()
              if key not in {"timeline", "_nand_numeric_enabled", "_nand_state_exposed"}}
    return deepcopy(values), deepcopy(vars(runtime.timeline))


def copied_published(runtime):
    values = deepcopy({key: value for key, value in vars(runtime.runtimes["hbf"].core).items()
                       if key not in {"timeline", "_nand_numeric_enabled", "_nand_state_exposed"}})
    packed = values["_nand_numeric_state"]
    if packed is not None:
        packed.publish(values["_read_cache"], values["_array_ready"], values["_buffer_ready"])
    values["_nand_numeric_state"] = None
    return values, deepcopy(vars(runtime.timeline))


def assert_results(a, b):
    assert np.array_equal(a[0], b[0])
    assert a[1:] == b[1:]


def test_resident_page_reads_writes_cache_lifetimes_and_scalar_handoffs_are_exact():
    reference, resident = contexts()
    answers = []
    for runtime in (reference, resident):
        core = runtime.runtimes["hbf"].core
        rows = []
        for operation, addresses in (("read", list(range(1024))), ("read", list(range(8))),
                                     ("write", list(range(4))), ("read", list(range(128)))):
            rows.append(getattr(core, "submit_" + operation + "_pages")(
                np.asarray(addresses, dtype=np.int64) * 4096, 0.))
        if runtime is resident:
            assert core._nand_numeric_state is not None
            assert not core.__dict__["_read_cache"]  # No per-batch Python publication.
        # A partial request uses the same live buffer tags and readers.
        rows.append(core.submit(AccessRequest("partial", "read", 64, 64)))
        rows.append(core.submit_write_pages([4 * 4096, 5 * 4096], [0., 1.]))
        rows.append(core.submit_read_pages([0, 4 * 4096, 5 * 4096], [0., 2., 3.]))
        answers.append(rows)
    for a, b in zip(*answers):
        if isinstance(a, tuple):
            assert_results(a, b)
        else:
            assert a == b
    assert published(reference) == published(resident)


@pytest.mark.parametrize("finish_inner", ["commit", "rollback"])
@pytest.mark.parametrize("resident_at_begin", [False, True])
def test_nested_numeric_and_scalar_rollback_restores_original_native_state(finish_inner, resident_at_begin):
    reference, resident = contexts()
    for runtime in (reference, resident):
        core = runtime.runtimes["hbf"].core
        if resident_at_begin:
            core.submit_read_pages(np.arange(128, dtype=np.int64) * 4096, 0.)
        outer = runtime.begin_transaction()
        core.submit(AccessRequest("scalar-first", "read", 0, 64))
        core.submit_read_pages(np.arange(128, dtype=np.int64) * 4096, 0.)
        inner = runtime.begin_transaction()
        core.submit_write_pages([0, 4096], 0.)
        core.submit(AccessRequest("scalar-inner", "read", 0, 64))
        getattr(runtime, "commit_transaction" if finish_inner == "commit" else "restore")(inner)
        core.submit_read_pages([0, 4096, 8192], 0.)
        runtime.restore(outer)
    assert published(reference) == published(resident)


@pytest.mark.parametrize("finish", ["commit", "rollback"])
def test_public_snapshot_reuses_active_resident_token_after_further_execution(finish):
    runtime = contexts()[1]
    core = runtime.runtimes["hbf"].core
    core.submit_read_pages(np.arange(128, dtype=np.int64) * 4096, 0.)
    core.submit_read_pages([0, 4096], 0.)
    expected = copied_published(runtime)
    outer = runtime.begin_transaction()
    core.submit_write_pages([0, 4096], 0.)
    saved = runtime.snapshot()
    for _ in range(2):
        core.submit_read_pages([0, 4096], 0.)
        getattr(runtime, "commit_transaction" if finish == "commit" else "restore")(outer)
        runtime.restore(saved)
        core.submit_read_pages([8192, 12288], 0.)
        runtime.restore(outer)
        assert copied_published(runtime) == expected
        runtime.restore(saved)


def test_exposed_mutable_state_disables_residency_and_is_protected_on_rollback():
    runtime = contexts()[1]
    core = runtime.runtimes["hbf"].core
    core.submit_read_pages([0, 4096], 0.)
    cache, arrays, buffers = core._read_cache, core._array_ready, core._buffer_ready
    assert core._nand_numeric_state is None and not core._nand_numeric_enabled
    before = deepcopy((cache, arrays, buffers))
    token = core.begin_transaction()
    next(iter(cache.values()))[0].ready_ns = 1e9
    arrays[next(iter(arrays))] = 1e9
    buffers[next(iter(buffers))] = 1e9
    core.submit_read_pages([0, 4096], 0.)
    core.rollback_transaction(token)
    assert (core._read_cache, core._array_ready, core._buffer_ready) == before


def test_numeric_failure_restores_resident_arrays_before_any_publication(monkeypatch):
    from heterollm_sim import _compiled_nand

    runtime = contexts()[1]
    core = runtime.runtimes["hbf"].core
    core.submit_read_pages([0, 4096], 0.)
    packed = core._nand_numeric_state
    before = packed.checkpoint()
    timeline = deepcopy(vars(runtime.timeline))
    execute = _compiled_nand._run
    def fail_after_mutation(*args):
        execute(*args)
        raise RuntimeError("after native cache mutation")
    monkeypatch.setattr(_compiled_nand, "_run", fail_after_mutation)
    with pytest.raises(RuntimeError, match="native cache"):
        core.submit_read_pages([8192, 12288], 0.)
    assert all(np.array_equal(a, b) for a, b in zip(before[0], packed.arrays))
    assert np.array_equal(before[1], packed.dirty_units)
    assert np.array_equal(before[2], packed.dirty_banks)
    assert vars(runtime.timeline) == timeline


def test_random_nested_transactions_match_published_page_machine():
    from random import Random

    reference, resident = contexts()
    rng = Random(6349)
    config = reference.runtimes["hbf"].core.config
    banks = (config.channels * config.targets_per_channel * config.dies_per_target
             * config.luns_per_die * config.planes_per_lun)
    for wave in range(30):
        outer = [runtime.begin_transaction() for runtime in (reference, resident)]
        inner = None
        for step in range(8):
            if step == 2:
                inner = [runtime.begin_transaction() for runtime in (reference, resident)]
            if step == 6:
                finish = "commit_transaction" if rng.randrange(2) else "restore"
                for runtime, token in zip((reference, resident), inner):
                    getattr(runtime, finish)(token)
            addresses = [rng.randrange(128) * 4096 for _ in range(rng.randrange(2, 10))]
            arrivals = [rng.randrange(1000) * .5 for _ in addresses]
            operation = "write" if rng.randrange(4) == 0 else "read"
            if operation == "write":
                addresses = [address % (banks * 4096) for address in addresses]
            outputs = []
            for runtime in (reference, resident):
                core = runtime.runtimes["hbf"].core
                if step == 4:
                    outputs.append(core.submit(AccessRequest("partial", "read", addresses[0] + 64, 64)))
                else:
                    outputs.append(getattr(core, "submit_" + operation + "_pages")(addresses, arrivals))
            if step == 4:
                assert outputs[0] == outputs[1]
            else:
                assert_results(*outputs)
        finish = "commit_transaction" if rng.randrange(2) else "restore"
        for runtime, token in zip((reference, resident), outer):
            getattr(runtime, finish)(token)
        # Observe independent copies, leaving the live native state resident
        # across the next transaction's begin/commit/rollback sequence.
        assert copied_published(reference) == copied_published(resident), wave


def test_gc_scalar_handoffs_retain_the_same_resident_arrays():
    """A scalar GC copy must not unload unrelated banks between page runs."""
    reference, resident = contexts()
    for runtime in (reference, resident):
        core = runtime.runtimes["hbf"].core
        core.submit_read_pages(np.arange(256, dtype=np.int64) * 4096, 0.)
    core = resident.runtimes["hbf"].core
    packed = core._nand_numeric_state
    arrays = packed.arrays
    for request in (AccessRequest("gc-read", "read", 128 * 4096, 4096),
                    AccessRequest("gc-program", "write", 0, 4096),
                    AccessRequest("partial-read", "read", 64, 64)):
        expected = reference.runtimes["hbf"].core.submit(request)
        actual = core.submit(request)
        assert actual == expected
        assert core._nand_numeric_state is packed
        assert all(current is original for current, original in zip(packed.arrays, arrays))
    # Only the two scalar banks need Python slot objects. All other banks
    # stay numeric until this explicit public-state comparison.
    assert len(core.__dict__["_read_cache"]) == 2
    assert published(reference) == published(resident)


def test_scalar_handoff_exception_retains_exact_partial_state():
    from heterollm_sim.memory_mapping import map_nand_address

    reference, resident = contexts()
    for runtime in (reference, resident):
        runtime.runtimes["hbf"].core.submit_read_pages(np.arange(32, dtype=np.int64) * 4096, 0.)
    # An overflow raised by the original scalar reserve follows its original
    # mutation ordering. The bridge publishes the same partial state later.
    request = AccessRequest("overflow", "read", 0, 64)
    for runtime in (reference, resident):
        core = runtime.runtimes["hbf"].core
        mapping = map_nand_address(core.config, request.address)
        core.timeline.busy_ns[core._host_id(mapping, request.operation)] = 10**400
        with pytest.raises(OverflowError, match="too large"):
            core.submit(request)
    assert published(reference) == published(resident)


@pytest.mark.parametrize("method", ["_cache_slot", "_write_buffer_payload_bytes"])
def test_authored_scalar_method_replacement_receives_fully_published_state(monkeypatch, method):
    from heterollm_sim.nand_core import NandCore

    runtime = contexts()[1]
    core = runtime.runtimes["hbf"].core
    core.submit_read_pages(np.arange(128, dtype=np.int64) * 4096, 0.)
    original = getattr(NandCore, method)
    observed = []
    def authored(self, *args):
        observed.append((self._nand_numeric_state, len(self.__dict__["_read_cache"])))
        return original(self, *args)
    monkeypatch.setattr(NandCore, method, authored)
    core.submit(AccessRequest("authored", "read", 0, 64))
    assert observed == [(None, 128)]


@pytest.mark.parametrize("method", ["_resident_scalar_eligible", "_submit_resident_scalar"])
@pytest.mark.parametrize("scope", ["instance", "class"])
def test_authored_resident_dispatch_cannot_enter_partially_published_state(monkeypatch, method, scope):
    from heterollm_sim.nand_core import NandCore

    runtime = contexts()[1]
    core = runtime.runtimes["hbf"].core
    core.submit_read_pages(np.arange(128, dtype=np.int64) * 4096, 0.)
    def authored(*args):
        pytest.fail("unqualified resident helper was called before full publication")
    monkeypatch.setattr(core if scope == "instance" else NandCore, method, authored)
    core.submit(AccessRequest("authored-resident-dispatch", "read", 0, 64))
    assert core._nand_numeric_state is None
    assert len(core.__dict__["_read_cache"]) == 128


def test_scalar_handoff_uses_the_same_multi_plane_bank_and_array():
    config = replace(resolve_physical_config("hbf_ocp_0_7"),
                     channels=2, host_channels=2, targets_per_channel=2, dies_per_target=2,
                     luns_per_die=2, planes_per_lun=2, capacity_bytes=None)
    reference, resident = [PhysicalRuntimeContext(config=config, physical_owner="hbf",
                                                  capture_details=False) for _ in range(2)]
    reference.runtimes["hbf"].core._nand_numeric_enabled = False
    base_units = config.channels * config.targets_per_channel * config.dies_per_target * config.luns_per_die
    outputs = []
    for runtime in (reference, resident):
        core = runtime.runtimes["hbf"].core
        core.submit_read_pages(np.arange(base_units * 2, dtype=np.int64) * 4096, 0.)
        rows = []
        for bank in (base_units + 1, 1, base_units * 2 - 1):
            rows.append(core.submit(AccessRequest("read-bank", "read", bank * 4096 + 64, 64)))
            rows.append(core.submit(AccessRequest("write-bank", "write", bank * 4096, 4096)))
        outputs.append(rows)
    assert outputs[0] == outputs[1]
    assert published(reference) == published(resident)


def test_scalar_cache_clock_beyond_native_range_keeps_python_integer_semantics():
    reference, resident = contexts()
    results = []
    for runtime in (reference, resident):
        core = runtime.runtimes["hbf"].core
        core.submit_read_pages([0, 4096], 0.)
        core._cache_use = 2**63 - 1
        results.append(core.submit(AccessRequest("large-clock", "read", 0, 64)))
        assert core._cache_use == 2**63
    assert results[0] == results[1]
    assert published(reference) == published(resident)


@pytest.mark.parametrize("changed", ["array_resource_prefix", "parallel_resource_prefix",
                                   "channel_resource_prefix", "host_resource_id", "resource_alias"])
def test_scalar_handoff_publishes_before_mutable_resource_layout_changes(changed):
    reference, resident = contexts()
    results = []
    for runtime in (reference, resident):
        core = runtime.runtimes["hbf"].core
        # Each context has its own metadata so the second oracle starts from
        # the same original resource identifiers.
        core.config = replace(core.config, metadata=dict(core.config.metadata))
        core.submit_read_pages(np.arange(128, dtype=np.int64) * 4096, 0.)
        if changed == "resource_alias":
            core.timeline.resource_aliases["nand:array:0"] = "shared:array"
        else:
            core.config.metadata[changed] = "changed:" + changed
        request = AccessRequest("changed-resources", "write", 0, 4096)
        if runtime is resident:
            assert not core._resident_scalar_eligible(request)
        rows = [core.submit(request)]
        rows.append(core.submit_read_pages([0, 4096], 0.))
        results.append(rows)
    assert results[0][0] == results[1][0]
    assert_results(results[0][1], results[1][1])
    assert published(reference) == published(resident)
