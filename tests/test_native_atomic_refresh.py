"""Atomic native descriptors share arrival, while Task boundaries remain."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from heterollm_sim.contracts import TaskSpec, TaskCategory, _PreparedExecutionStage
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import NandConfig
from heterollm_sim.native_window_producer import (_atomic_task_refresh_bound,
    _refresh_serial_terms, summarize_native_stages)
from test_hbm_compiled_control import configuration
from test_native_window_producer import task


def settings():
    hbm = configuration(refresh_policy="all_bank", refresh_interval_ns=3900., refresh_cycle_ns=40.)
    nand = NandConfig(kind="HBF", page_bytes=4096, page_read_ns=1e6, capacity_bytes=2**20)
    return hbm, nand


def group(metadata=None, *, nand_changes=None, aliases=None):
    hbm, nand = settings()
    if nand_changes:
        nand = replace(nand, **nand_changes)
    row = task("mixed", 0, metadata=metadata)
    terms = [_refresh_serial_terms(hbm, 32, "read", independent_channels=True)]*3
    old = sum((1+int(d//(interval-blocked)))*blocked for interval, blocked, d in terms)
    physical = NS(timeline=ResourceTimeline())
    physical.timeline.resource_aliases.update(aliases or {})
    value, proof = _atomic_task_refresh_bound(row, {"hbm": terms}, {"hbm": hbm, "nand": nand},
        physical, old, independent_channels=True)
    return value, proof, old


def test_mixed_native_task_pays_one_initial_REF_and_keeps_complete_active_feedback():
    value, proof, old = group()
    assert value < old
    assert proof["descriptor_count"] == 3 and not proof["between_Task_waits_removed"]
    assert proof["serialized_native_no_refresh_service_upper_ns"] > 0.


@pytest.mark.parametrize("metadata", ({"hbf_host_staging_by_owner": {"hbm": {}}},
    {"stateful_l2": {"lookup_ns": 1.}}, {"l2_execution": {"backing_arrival_offset_ns": 10.}}))
def test_staging_or_L2_arrival_paths_do_not_share_the_atomic_REF_envelope(metadata):
    value, proof, old = group(metadata)
    assert value == old and proof is None


@pytest.mark.parametrize("metadata", (
    {"event_kind": "kv_prefetch"},
    {"kv_cache_component": "hbf"},
    {"kv_operand": "k"},
    {"kv_read_semantics": "external_persisted_kv_cache_only"},
    {"native_kv_work": {}},
    {"source_f32_kv": {}},
    {"persistent_request_buffers": {"r0": ("cache:k", "cache:v")}},
    {"persistent_buffer_ids": ("native_kv:r0:layer0:rank0:k",)},
    {"memory_accesses": ({"buffer_id": "kv:r0:layer0:rank0:v"},)},
    {"physical_allocations": ({"buffer_id": "native_kv:r0:layer0:rank0:k"},)},
    {"buffer_accesses": ({"buffer_id": "r0.layer0.k_cache"},)},
    {"target_buffer_accesses": ({"buffer_id": "r0.layer0.v_cache"},)},
))
def test_KV_descriptors_without_generic_metadata_keep_per_access_future_REF_protection(metadata):
    value, proof, old = group(metadata)
    assert value == old and proof is None


def test_direct_and_aliased_NAND_HBM_shared_bus_prevent_atomic_epoch_sharing():
    for changes, aliases in (({"metadata": {"host_resource_id": "dram:row_command:0:0"}}, None),
            (None, {"nand:host": "dram:row_command:0:0"})):
        value, proof, old = group(nand_changes=changes, aliases=aliases)
        assert value == old and proof is None


def test_real_native_mixed_task_keeps_NAND_time_out_of_HBM_arrival_and_preserves_RAW(monkeypatch):
    hbm, nand = settings()
    calls = []
    original = DramCore.submit
    def record(self, request, **kwargs):
        result = original(self, request, **kwargs)
        calls.append((request.operation.value, request.address, request.arrival_ns, result.completion_ns))
        return result
    monkeypatch.setattr(DramCore, "submit", record)
    accesses = (
        {"physical_owner": "hbm", "resource_id": "hbm", "operation": "read", "address": 0, "byte_count": 32},
        {"physical_owner": "nand", "resource_id": "nand", "operation": "read", "address": 0, "byte_count": 4096},
        {"physical_owner": "hbm", "resource_id": "hbm", "operation": "write", "address": 0, "byte_count": 32},
        {"physical_owner": "hbm", "resource_id": "hbm", "operation": "read", "address": 4096, "byte_count": 32})
    native_task = TaskSpec("mixed", "r0", "mixed", TaskCategory.MEMORY, (), (), metadata={
        "physical_owner": "hbm", "physical_memory_config": hbm,
        "physical_memory_configs": {"hbm": hbm, "nand": nand}, "memory_accesses": accesses})
    context = PhysicalRuntimeContext(hbm, "hbm", capture_details=False)
    result = resolve_physical_task(native_task, context, 0.)
    assert [row[2] for row in calls] == [0., calls[0][3], 0.]
    assert result.metadata["physical_execution_by_owner"]["nand"]["completion_ns"] >= 1e6
    assert max(row[2] for row in calls) < 1e6


def test_summary_records_atomic_proof_without_removing_between_Task_boundaries():
    hbm, nand = settings()
    metadata = {"physical_owner": "hbm", "physical_memory_config": hbm,
        "physical_memory_configs": {"hbm": hbm, "nand": nand},
        "memory_accesses": tuple({"physical_owner": owner, "resource_id": owner,
            "operation": "read", "byte_count": 32 if owner == "hbm" else 4096}
            for owner in ("hbm", "nand", "hbm", "hbm"))}
    first = task("a", 0, metadata=metadata)
    second = task("b", 0, ("a",), metadata=metadata)
    stage = _PreparedExecutionStage("body", 0, (), ("r0",), "gpu0", 0., (first, second))
    context = NS(timeline=ResourceTimeline(), runtimes={}, allocators={})
    summary = summarize_native_stages((stage,), NS(), context)
    proofs = summary["atomic_task_refresh_proofs"]
    assert len(proofs) == 2 and {row["task_id"] for row in proofs} == {"a", "b"}
    assert all(not row["between_Task_waits_removed"] for row in proofs)
    assert summary["owner_accesses"]["hbm"]["refresh_active_io_serial_upper_ns"] > sum(
        row["atomic_task_refresh_upper_ns"] for row in proofs)
