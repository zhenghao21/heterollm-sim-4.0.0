"""HBM page lifetime and executed migration accounting across runtime layers."""
from dataclasses import replace
from heterollm_sim.data_motion import lower_physical_allocations

import pytest

from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
from heterollm_sim.memory_types import make_hbf_config, make_hbm_config
from test_kv_transfer_deep_semantics import component, page, request
from test_backend_page_capacity_followup import fixture, allocate


def test_public_page_access_export_reuses_core_allocation_after_handoff_and_restore():
    from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
    from heterollm_sim.data_motion import lower_physical_allocations, resolve_physical_task
    pool, executor, _ledger = fixture()
    (item,) = allocate(pool, executor, 1)
    initial = executor._binding(item).allocation
    pool.register_prefix("shared", (item,))
    assert pool.resize_detailed("reader", 1, prefix_key="shared").success
    executor.sync_pages(pool.pages())
    assert pool.cache_accesses("reader", first_token=0, token_count=16) == pool.cache_accesses(
        "request", first_token=0, token_count=16)
    pool.release("reader")
    pool.release_prefix("shared")
    pool.handoff_request("request", "child")
    executor.sync_pages(pool.pages())
    for migrated in (False, True):
        if migrated:
            assert pool.offload_page(item.logical_page_id, "hbf")
            assert pool.restore_page(item.logical_page_id, "hbm")
        (component_id, access), = pool.cache_accesses("child", first_token=0, token_count=16)
        binding = executor._binding(item)
        assert access.buffer_id == binding.allocation.buffer_id
        assert access.allocation_generation == binding.generation == (2 if migrated else 0)
        assert access.buffer_size_bytes == binding.allocation.size_bytes
        if not migrated:
            assert access.buffer_id == initial.buffer_id  # Handoff does not rename physical storage.
        before = executor.runtime.allocators[binding.owner].allocations()
        task = TaskSpec("consumer", "child", "kv-read", TaskCategory.MEMORY,
            demands=(ResourceDemand(binding.owner, 0., bytes_moved=access.size_bytes),),
            metadata={"physical_owner": binding.owner,
                "physical_memory_config": executor._configs[component_id],
                "physical_external_allocations": ((binding.owner, access.buffer_id, binding.generation),),
                "memory_accesses": ({"buffer_id": access.buffer_id, "operation": access.operation,
                    "offset_bytes": access.offset_bytes, "byte_count": access.size_bytes,
                    "allocation_generation": access.allocation_generation,
                    "allocation_size_bytes": access.buffer_size_bytes},)})
        resolved = resolve_physical_task(lower_physical_allocations(task), executor.runtime, 0.)
        assert resolved.metadata["memory_accesses"][0]["address"] == binding.allocation.base_address
        assert executor.runtime.allocators[binding.owner].allocations() == before


def test_cache_rollback_restores_lru_and_subsequent_actual_hbm_cost():
    from dataclasses import asdict
    from heterollm_sim.cache_state import CacheAccess, ExplicitCacheState
    from heterollm_sim.event_kernel import UnifiedEventKernel
    from test_native_kernel_physical_cost import native_task

    def run(*, abort):
        cache = ExplicitCacheState(256, 64)
        cache.read("0", 0, 64, buffer_size_bytes=128)  # Clean oldest line.
        for name in ("1", "2", "3"):
            cache.write(name, 0, 64, buffer_size_bytes=128)
        before = cache.snapshot(), cache.resident_lines()
        if abort:
            transaction = cache.begin_transaction()
            cache.write("4", 64, 64, buffer_size_bytes=128)
            cache.flush("3")
            cache.write("1", 64, 64, buffer_size_bytes=128)
            cache.write("0", 64, 64, buffer_size_bytes=128)
            cache.write("5", 0, 64, buffer_size_bytes=128)
            cache.rollback_transaction(transaction)
            assert (cache.snapshot(), cache.resident_lines()) == before
        task, _ = native_task(1000., l2=True)
        contract = {**task.metadata["stateful_l2"], "capacity_bytes": 256,
                    "accesses": (asdict(CacheAccess.read("new", 0, 64, buffer_size_bytes=128)),)}
        task = replace(task, metadata={**task.metadata, "physical_allocations": (), "stateful_l2": contract,
            "memory_accesses": ({"buffer_id": "new", "operation": "read", "offset_bytes": 0,
                               "byte_count": 64, "allocation_size_bytes": 128},)})
        kernel = UnifiedEventKernel.from_closed_graph((lower_physical_allocations(task),))
        kernel._l2_states["gpu.l2"] = ((256, 64, True, True), cache)
        event = kernel.step()
        assert event.task.metadata["l2_execution"]["dirty_eviction_bytes"] == 0
        physical = next(iter(event.task.metadata["physical_execution_by_owner"].values()))
        assert physical["physical_read_bytes"] == 64
        assert physical["physical_write_bytes"] == 0
        return event.demands, event.end_ns, physical

    assert run(abort=True) == run(abort=False)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_migrated_kv_generation_stays_resident_with_live_workspace(asynchronous):
    # The page exceeds the entire workspace; neither a live activation nor a
    # new KV generation may move it out of its independent resident arena.
    hbm = make_hbm_config(capacity_bytes=32768)
    hbm = replace(hbm, metadata={**hbm.metadata, "workspace_capacity_bytes": 4096})
    hbf = make_hbf_config(capacity_bytes=65536, page_bytes=16384,
                          partial_page_policy="read_modify_write")
    executor = PhysicalKVTransferExecutor([component("hbm", hbm), component("hbf", hbf)])
    item = page("hbm", 8192)
    executor.bind_page(item)
    hbm_owner = executor.services["hbm"].physical_owner
    allocator = executor.runtime.allocators[hbm_owner]
    activation = allocator.allocate("live-activation", 4096, 1)
    assert activation.base_address == 28672
    for generation, target in enumerate(("hbf", "hbm", "hbf", "hbm"), 1):
        plan = executor.prepare_async(request(item, target)) if asynchronous else executor(request(item, target))
        while plan.advance is not None:
            plan = plan.advance()
        plan.commit()
        item.owner_component = target
        item.allocation_generation = generation
        binding = executor._binding(item)
        assert binding.generation == generation
        assert binding.allocation.allocation_arena == "resident"
        if target == "hbm":
            assert binding.allocation.base_address + binding.allocation.size_bytes <= 28672
            # Snapshot restore must retain the class independently of version.
            executor.restore(executor.snapshot())
            allocator = executor.runtime.allocators[hbm_owner]
            assert allocator.get_allocation("live-activation", 1) == activation
    bill = executor.cost_ledger()
    assert bill["owners"][hbm_owner]["physical_read_bytes"] == 2 * 8192
    assert bill["owners"][hbm_owner]["physical_write_bytes"] == 2 * 8192
    executor.release_page(item)
    assert executor.runtime.allocators[hbm_owner].allocations() == (activation,)


@pytest.mark.parametrize("details", [False, True])
def test_hbm_snapshot_isolates_all_mutable_histories_and_repeated_restore(details):
    from copy import deepcopy
    from heterollm_sim.data_motion import PhysicalRuntimeContext
    from heterollm_sim.memory_types import AccessRequest
    from test_hbm_compiled_control import configuration

    config = configuration(refresh_policy="all_bank", refresh_interval_ns=400.,
        refresh_cycle_ns=40., four_activate_window_ns=20. if details else 0.)
    runtime = PhysicalRuntimeContext(config, "memory", capture_details=details)
    core = runtime.runtimes["memory"].core
    core.submit(AccessRequest("seed-read", "read", 0, 4096))
    core.submit(AccessRequest("seed-write", "write", 0, 1024, 200.))
    state = core._hbm_control_state
    bank_id = next(iter(state["banks"]))
    # Unknown nested fields must remain isolated too, including their aliases
    # to another part of the controller state.
    extension = {"samples": [1., {"times": [2.]}]}
    state["banks"][bank_id]["extension"] = extension
    state["extension"] = extension
    expected = deepcopy(state)
    saved = runtime.snapshot()
    saved_state = saved["runtime_state"]["memory"]["core"]["_hbm_control_state"]

    def mutable_ids(value):
        if isinstance(value, dict):
            return {id(value)}.union(*(mutable_ids(item) for item in value.values()))
        if isinstance(value, list):
            return {id(value)}.union(*(mutable_ids(item) for item in value))
        return set()

    assert saved_state == expected
    assert mutable_ids(saved_state).isdisjoint(mutable_ids(state))
    assert saved_state["extension"] is saved_state["banks"][bank_id]["extension"]
    assert saved_state["banks"][bank_id]["mapping"] is state["banks"][bank_id]["mapping"]
    outcomes = []
    for _ in range(2):
        result = core.submit(AccessRequest("after-refresh", "read", 8192, 8192, 5000.))
        core._hbm_control_state["extension"]["samples"][1]["times"].append(99.)
        assert core._hbm_control_state != expected
        assert saved_state == expected
        outcomes.append((result, deepcopy(core._hbm_control_state), deepcopy(core._banks),
                         deepcopy(runtime.timeline.ready_ns)))
        runtime.restore(saved)
        assert core._hbm_control_state == expected
        assert mutable_ids(saved_state).isdisjoint(mutable_ids(core._hbm_control_state))
        assert runtime.snapshot() == saved
    assert outcomes[0] == outcomes[1]


def test_hbm_history_copy_preserves_cycles_and_cross_container_aliases():
    from heterollm_sim.data_motion import _copy_hbm_control_state

    samples = [1.]
    samples.append((samples,))
    bank = {"last_activate": 2., "extension": samples}
    state = {"banks": {"first": bank, "alias": bank},
             "pcs": {(0, 0, 0): {"activate_window": samples}}, "extension": samples}
    state["self"] = state
    cloned = _copy_hbm_control_state(state)
    assert cloned is not state
    assert cloned["self"] is cloned
    assert cloned["banks"]["first"] is cloned["banks"]["alias"]
    copied = cloned["extension"]
    assert copied is not samples
    assert copied[1][0] is copied
    assert cloned["banks"]["first"]["extension"] is copied
    assert cloned["pcs"][(0, 0, 0)]["activate_window"] is copied
    copied.append(99.)
    assert len(samples) == 2


def test_hbm_history_copy_keeps_extension_deepcopy_protocol_and_order():
    from heterollm_sim.data_motion import _copy_hbm_control_state

    calls = []

    class Extension:
        def __init__(self, label):
            self.label = label

        def __deepcopy__(self, memo):
            calls.append(self.label)
            return Extension(self.label)

    key, value = Extension("key"), Extension("value")
    state = {"banks": {}, "extension": {key: value}, "alias": value}
    cloned = _copy_hbm_control_state(state)
    # deepcopy(dict) evaluates the value before the key and invokes an
    # extension object's protocol only once, even across shared references.
    assert calls == ["value", "key"]
    copied_key, copied_value = next(iter(cloned["extension"].items()))
    assert copied_key is not key and copied_value is not value
    assert copied_key.label == "key" and copied_value.label == "value"
    assert cloned["alias"] is copied_value


def test_hbm_history_copy_honors_extension_memo_entries_for_scalars_and_tuple_keys():
    from heterollm_sim.data_motion import _copy_hbm_control_state

    scalar = int("12345")

    class MemoExtension:
        def __deepcopy__(self, memo):
            memo[id(scalar)] = "replacement"
            return "copied extension"

    state = {"banks": {}, "extension": MemoExtension(),
             "history": {(scalar, 2): [scalar], "scalar": scalar, scalar: scalar}}
    cloned = _copy_hbm_control_state(state)
    assert cloned["extension"] == "copied extension"
    assert cloned["history"] == {("replacement", 2): ["replacement"],
                                 "scalar": "replacement", "replacement": "replacement"}


@pytest.mark.parametrize("details", [False, True])
def test_failed_hbm_execution_restores_control_state_before_retry(details, monkeypatch):
    from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
    from heterollm_sim.dram_core import DramCore
    from test_hbm_compiled_control import configuration
    from test_hbm_protocol_accounting import physical_task

    config = configuration(four_activate_window_ns=20. if details else 0.)
    runtime = PhysicalRuntimeContext(capture_details=details)
    seed = physical_task(config, [{"operation": "write", "address": 0, "byte_count": 4096}])
    resolve_physical_task(seed, runtime, 0.)
    saved = runtime.snapshot()
    task = physical_task(config, [{"operation": "read", "address": 8192, "byte_count": 4096}])
    original = DramCore.submit

    def fail_after_execution(core, request):
        original(core, request)
        raise ValueError("injected after HBM controller mutation")

    with monkeypatch.context() as patch:
        patch.setattr(DramCore, "submit", fail_after_execution)
        with pytest.raises(ValueError, match="after HBM controller mutation"):
            resolve_physical_task(task, runtime, 0.)
    assert runtime.snapshot() == saved
    expected = resolve_physical_task(task, runtime, 0.)
    runtime.restore(saved)
    assert resolve_physical_task(task, runtime, 0.) == expected
