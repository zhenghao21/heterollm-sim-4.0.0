"""Generic NAND reuses the original private controller and scalar page machine."""
from copy import deepcopy
from dataclasses import replace

import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_types import AccessRequest
from test_nand_generic_buffered_acceleration import config, PAGE
from test_token_owned_history import bits


def pair(settings=None):
    settings = settings or config()
    contexts = [PhysicalRuntimeContext(settings, "flash", capture_details=False) for _ in range(2)]
    contexts[1].runtimes["flash"].core._nand_numeric_enabled = False
    return contexts


def logical(core):
    copy = deepcopy(core)
    copy._nand_publish_numeric()
    return {name: copy.__dict__[name] for name in (
        "_array_ready", "_buffer_ready", "_read_cache", "_cache_use", "_next_program_page",
        "_block_wear_state", "_inflight", "_inflight_operations", "_acceptance_ns",
        "_host_inflight", "_host_acceptance_ns", "_write_buffer_pending",
        "_write_buffer_occupancy", "_staged_pages", "_buffer_merge_bytes",
        "_media_drain_completion_ns", "_observation_events")}, {
        name: value for name, value in vars(copy.timeline).items()
        if name not in ("_owned_transactions", "_metric_key_orders")}


def same(contexts):
    assert bits(logical(contexts[0].runtimes["flash"].core)) == bits(logical(contexts[1].runtimes["flash"].core))


def submit(contexts, request):
    actual, expected = [context.runtimes["flash"].core.submit(request) for context in contexts]
    assert bits(actual) == bits(expected)
    same(contexts)
    return actual


@pytest.mark.parametrize("independent", (False, True))
@pytest.mark.parametrize("duplex", (False, True))
def test_bulk_and_original_single_page_machine_retain_private_arrays(independent, duplex):
    contexts = pair(config(planes_independent=independent, host_full_duplex=duplex))
    submit(contexts, AccessRequest("seed", "read", 0, 256 * PAGE, .125))
    core = contexts[0].runtimes["flash"].core
    packed = core._nand_numeric_state
    assert packed is not None
    for index, (operation, address, count) in enumerate((
            ("read", 255 * PAGE + 17, 111),
            ("write", 255 * PAGE + 33, 90),
            ("read", 255 * PAGE + 17, 111),
            ("erase", 0, PAGE),
            ("write", 0, PAGE),
            ("read", 256 * PAGE, 384 * PAGE))):
        submit(contexts, AccessRequest(str(index), operation, address, count, 10000.125 + index))
        assert core._nand_numeric_state is packed
    # A short unaligned span uses the unchanged detailed path against only
    # its addressed banks, keeping the remaining controller private.
    submit(contexts, AccessRequest("boundary", "write", PAGE - 7, 16, 20000.25))
    assert core._nand_numeric_state is packed
    submit(contexts, AccessRequest("bulk-again", "read", 0, 384 * PAGE, 30000.25))
    assert core._nand_numeric_state is not None


@pytest.mark.parametrize("inner_owned", (False, True))
@pytest.mark.parametrize("finish", ("commit", "restore"))
def test_resident_generic_nested_runtime_rollback_is_exact(inner_owned, finish):
    contexts = pair()
    submit(contexts, AccessRequest("seed", "read", 0, 256 * PAGE, .125))
    original = bits(logical(contexts[0].runtimes["flash"].core))
    outer = [context.begin_transaction(owners=("flash",), _owned_timeline=index == 0)
             for index, context in enumerate(contexts)]
    assert outer[0]["runtime_state"]["flash"]["nand_transaction"].array_frontiers is None
    submit(contexts, AccessRequest("prefix", "write", 256 * PAGE, 256 * PAGE, 20000.25))
    prefix = bits(logical(contexts[0].runtimes["flash"].core))
    inner = [context.begin_transaction(owners=("flash",), _owned_timeline=index == 0 and inner_owned)
             for index, context in enumerate(contexts)]
    submit(contexts, AccessRequest("small", "read", PAGE + 17, 101, 30000.25))
    submit(contexts, AccessRequest("bulk", "read", 512 * PAGE, 256 * PAGE, 40000.25))
    for context, token in zip(contexts, inner):
        context.restore(token)
    same(contexts)
    assert bits(logical(contexts[0].runtimes["flash"].core)) == prefix
    for context, token in zip(contexts, outer):
        getattr(context, "commit_transaction" if finish == "commit" else "restore")(token)
    same(contexts)
    if finish == "restore":
        assert bits(logical(contexts[0].runtimes["flash"].core)) == original
    submit(contexts, AccessRequest("after", "read", 256 * PAGE, 256 * PAGE, 50000.25))


def test_public_cache_alias_publishes_once_and_remains_live():
    contexts = pair()
    submit(contexts, AccessRequest("seed", "read", 0, 256 * PAGE))
    aliases = [context.runtimes["flash"].core._read_cache for context in contexts]
    slots = [next(iter(cache.values()))[0] for cache in aliases]
    for slot in slots:
        slot.reader_completion_ns += 12345.125
    submit(contexts, AccessRequest("after-edit", "read", 0, 256 * PAGE, 10000.))
    for context, alias, slot in zip(contexts, aliases, slots):
        core = context.runtimes["flash"].core
        assert core._nand_numeric_state is None
        assert core.__dict__["_read_cache"] is alias
        assert next(iter(alias.values()))[0] is slot


def test_public_alias_materializes_numeric_transaction_preimage_before_external_edit():
    contexts = pair()
    submit(contexts, AccessRequest("seed", "read", 0, 256 * PAGE))
    previous = bits(logical(contexts[0].runtimes["flash"].core))
    tokens = [context.begin_transaction(owners=("flash",), _owned_timeline=index == 0)
              for index, context in enumerate(contexts)]
    submit(contexts, AccessRequest("prefix", "write", 256 * PAGE, 256 * PAGE, 20000.25))
    for context in contexts:
        core = context.runtimes["flash"].core
        core._array_ready["authored"] = 12345.125
        next(iter(core._read_cache.values()))[0].reader_completion_ns += 45678.25
    for context, token in zip(contexts, tokens):
        context.restore(token)
    same(contexts)
    assert bits(logical(contexts[0].runtimes["flash"].core)) == previous


def test_buffered_write_controller_keeps_its_original_shared_scalar_state():
    contexts = pair(config(write_buffer_bytes=1024 * 1024))
    submit(contexts, AccessRequest("read", "read", 0, 256 * PAGE))
    assert contexts[0].runtimes["flash"].core._nand_numeric_state is None
    submit(contexts, AccessRequest("write", "write", 256 * PAGE, 256 * PAGE, 20000.25))


def test_unknown_scalar_callback_observes_complete_published_cache():
    contexts = pair()
    submit(contexts, AccessRequest("seed", "read", 0, 256 * PAGE))
    observed = [[], []]
    for index, context in enumerate(contexts):
        core = context.runtimes["flash"].core
        original = core._execute_detailed_accepted
        def callback(request, original=original, core=core, index=index):
            observed[index].append(bits(core.__dict__["_read_cache"]))
            return original(request)
        core._execute_detailed_accepted = callback
    submit(contexts, AccessRequest("authored", "read", 17, 99, 10000.))
    assert observed[0] == observed[1]
    assert contexts[0].runtimes["flash"].core._nand_numeric_state is None


def test_changed_resource_prefix_publishes_old_controller_before_rebinding():
    contexts = pair()
    submit(contexts, AccessRequest("seed", "read", 0, 256 * PAGE))
    for context in contexts:
        core = context.runtimes["flash"].core
        core.config = replace(core.config, metadata={**core.config.metadata, "array_resource_prefix": "changed"})
    submit(contexts, AccessRequest("rebind", "read", 0, 256 * PAGE, 10000.))


def test_large_geometry_bulk_decline_hands_only_touched_banks_to_original_scalar_path():
    contexts = pair(config(channels=7, dies_per_target=16, targets_per_channel=1, planes_per_lun=25))
    submit(contexts, AccessRequest("seed", "read", 0, 4096 * PAGE))
    packed = contexts[0].runtimes["flash"].core._nand_numeric_state
    assert packed is not None
    submit(contexts, AccessRequest("declined", "read", 0, 256 * PAGE, 10000.))
    assert contexts[0].runtimes["flash"].core._nand_numeric_state is packed
