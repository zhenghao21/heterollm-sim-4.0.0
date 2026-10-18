"""Bind logical HBF page staging to real host DRAM and fabric calendars.

The NAND core continues to receive only complete, sequential OCP page writes.
Host acceptance is explicitly a host-memory operation, never a NAND completion.
"""
from __future__ import annotations

from .memory_types import DramConfig
from .physical_contract import parse_execution_physical_config


def configure(runtime, owner, config, binding):
    from .hbf_staging import HbfStagingController

    host_config = parse_execution_physical_config(binding["host_config"])
    host_owner = binding["host_owner"]
    capacity = binding["capacity_bytes"]
    if not isinstance(host_config, DramConfig) or host_config.kind.value == "HBM":
        raise ValueError("HBF host staging requires physical HOST DRAM")
    if host_owner == owner:
        raise ValueError("HBF staging and media require distinct owners")
    if binding.get("reserve_blocks", 1) != 1:
        raise ValueError("HBF host staging reserves exactly one physical erase block")
    identity = (host_owner, host_config, capacity)
    current = runtime.hbf_staging.get(owner)
    if current is not None:
        if current["identity"] != identity or current["config"] != config:
            raise ValueError("conflicting HBF host staging configuration for " + owner)
        return current
    runtime.runtime(config, owner)
    runtime.runtime(host_config, host_owner)
    controller = HbfStagingController(config, capacity)
    allocation = runtime.allocators[host_owner].allocate(
        "@hbf-host-staging:" + owner, capacity, allocation_arena="resident")
    current = {"identity": identity, "config": config, "controller": controller,
               "host_base": allocation.base_address, "host_owner": host_owner,
               "host_config": host_config, "preloaded": set()}
    runtime.hbf_staging[owner] = current
    return current


def preload(runtime, owner, buffer_id, generation, binding):
    state = runtime.hbf_staging[owner]
    allocator = runtime.allocators[owner]
    buffer_id, generation, _ = allocator.canonical_range(buffer_id, 0, 1, generation)
    key = (buffer_id, generation)
    if key in state["preloaded"]:
        return
    allocation = allocator.lookup(buffer_id, generation)
    controller = state["controller"]
    controller.preload(buffer_id, generation, allocation.size_bytes)
    # Warm weights were loaded before the measured request. Preserve their
    # physical program frontier so later writes cannot overwrite them.
    from .memory_mapping import map_nand_address
    core = runtime.runtimes[owner].core
    for address, count in controller.preload_page_counts(buffer_id, generation):
        mapping = map_nand_address(core.config, address)
        block = core._block_key(mapping)
        core._set_program_frontier(block, max(core._next_program_page.get(block, 0), count))
    state["preloaded"].add(key)


def release(runtime, owner, buffer_id, generation, arrival_ns=0.0):
    state = runtime.hbf_staging.get(owner)
    if state is not None:
        state["controller"].release(buffer_id, generation, arrival_ns)
        state["preloaded"].discard((buffer_id, generation))


def snapshot(states, transactional=False):
    return {owner: {**state, "preloaded": set(state["preloaded"]),
        "saved_controller": state["controller"].begin_transaction() if transactional else state["controller"].snapshot(),
        "controller_transaction": transactional} for owner, state in states.items()}


def commit(saved):
    for value in saved.values():
        if value["controller_transaction"]:
            value["controller"].commit_transaction(value["saved_controller"])


def restore(saved):
    states = {}
    for owner, value in saved.items():
        state = {key: item for key, item in value.items()
                 if key not in {"saved_controller", "controller_transaction"}}
        state["preloaded"] = set(value["preloaded"])
        if value["controller_transaction"]:
            state["controller"].rollback_transaction(value["saved_controller"])
        else:
            state["controller"].restore(value["saved_controller"])
        states[owner] = state
    return states
