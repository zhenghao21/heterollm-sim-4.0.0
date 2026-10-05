from dataclasses import asdict

import pytest

from heterollm_sim.cache_state import CacheAccess
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.cost_models import HBMProfile
from heterollm_sim.kernel_memory import resolve_l2_task


class _Allocator:
    """Small physical adapter matching the runtime allocator contract."""

    def __init__(self):
        self.rows = {
            ("root", 3): {"buffer_id": "root", "size_bytes": 128, "generation": 3, "base_address": 4096},
            ("alias", 3): {"buffer_id": "alias", "size_bytes": 16, "generation": 3,
                            "base_address": 4112, "alias_of": "root"},
            ("other", 0): {"buffer_id": "other", "size_bytes": 64, "generation": 0, "base_address": 8192},
        }

    def get_allocation(self, buffer_id, generation=0):
        return self.rows[(buffer_id, generation)]

    def canonical_range(self, buffer_id, offset, size, generation):
        row = self.get_allocation(buffer_id, generation)
        if offset + size > row["size_bytes"]:
            raise ValueError("range exceeds allocation")
        if row.get("alias_of"):
            return row["alias_of"], generation, 16 + offset
        return buffer_id, generation, offset


def _task(task_id, accesses, *, capacity=64, declarations=(), physical=True):
    hbm = HBMProfile(
        bandwidth_gb_s=100.0, read_bandwidth_gb_s=100.0, write_bandwidth_gb_s=100.0,
        resource_id="dram", transaction_bytes=64,
    )
    contract = {
        "owner": "gpu.l2", "capacity_bytes": capacity, "line_bytes": 64,
        "write_back": True, "write_allocate": True, "cache_resource": "l2",
        "memory_resource": "dram", "cache_bandwidth_gb_s": 1000.0,
        "cache_latency_ns": 1.0, "cache_parallelism": 1, "bandwidth_gb_s": 100.0,
        "hbm": asdict(hbm), "accesses": tuple(asdict(a) for a in accesses),
        "invocation_buffers": (), "envelope_resources": (), "order_resource": "l2.order",
        "dependency_ns": 0.0,
    }
    metadata = {"stateful_l2": contract, "physical_allocations": list(declarations)}
    if physical:
        metadata["physical_memory_config"] = {"kind": "dram"}
        metadata["physical_owner"] = "dram"
    return TaskSpec(
        task_id=task_id, request_id="r", name=task_id, category=TaskCategory.MEMORY,
        demands=(ResourceDemand("dram", 0), ResourceDemand("l2", 0)), metadata=metadata,
    )


def test_alias_is_canonicalized_before_cache_and_declarations_are_preserved():
    declarations = (
        {"buffer_id": "root", "size_bytes": 128, "generation": 3, "base_address": 4096},
        {"buffer_id": "alias", "size_bytes": 16, "generation": 3, "base_address": 4112,
         "alias_of": "root"},
    )
    task = _task("alias-read", (CacheAccess("alias", 0, 8, "read", 16, 3),), capacity=128, declarations=declarations)
    resolved = resolve_l2_task(task, {}, _Allocator())
    report = resolved.metadata["l2_execution"]
    assert report["accesses"][0]["buffer_id"] == "root"
    assert report["accesses"][0]["offset_bytes"] == 16
    assert report["physical_allocations"] == list(declarations)
    assert resolved.metadata["physical_allocations"] == [dict(item, physical_owner="dram") for item in declarations]


def test_partial_cold_fill_and_dirty_victim_are_reported():
    states = {}
    first = _task("first", (CacheAccess("root", 1, 1, "write", 64, 0),), physical=False)
    resolve_l2_task(first, states)
    second = _task("second", (CacheAccess("other", 0, 1, "read", 64, 0),), physical=False)
    result = resolve_l2_task(second, states)
    report = result.metadata["l2_execution"]
    assert report["hbm_read_bytes"] == 64
    assert report["hbm_write_bytes"] == 64
    assert report["dirty_eviction_bytes"] == 64


def test_duplicate_declarations_with_different_fixed_addresses_fail():
    declarations = (
        {"buffer_id": "root", "size_bytes": 64, "generation": 0, "base_address": 0},
        {"buffer_id": "root", "size_bytes": 64, "generation": 0, "base_address": 128},
    )
    task = _task("duplicate", (CacheAccess("root", 0, 1, "read"),), declarations=declarations)
    with pytest.raises(ValueError, match="conflicting physical allocation"):
        resolve_l2_task(task, {})
