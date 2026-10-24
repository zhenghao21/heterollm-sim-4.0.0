"""Row-state bounds for the boundary-dependent DRAM accesses of a sample window.

A native stationary sample window runs on a kernel copy taken at the first
cohort boundary. The open rows, recovery deadlines and activation history
every DRAM bank carries into the window are the boundary's, not the ones a
steady-state step inherits from the step before it. Inside a window each
later access to a bank sees the row state the window itself left, so only
the first access to each bank depends on the boundary. ``first_access_ns``
bounds, per sampled term, how far the term can move when every such access
takes any outcome between a row hit on a ready bank and a conflict behind
a row activated just before the access.

The bound propagates in max-plus form and assumes the executed service
order of every resource is kept. Each DRAM command path serves its bursts
first come, first served, so a burst waiting for its bank's row holds every
later burst of the path: a task's own delay is the largest per-path sum of
its first-access penalties. Delays then add along executed dependencies and
along the executed order of each resource a task demands (DRAM command and
data paths are task demands). A term starts behind a layer barrier that
waits for all earlier work, so terms are bounded separately.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .memory_types import DramConfig


def first_access_penalty_ns(config: DramConfig) -> float:
    """Largest row-outcome difference of a bank's first access in a window.

    The best case is a row hit on a ready bank; the worst case closes a row
    that was activated just before the access and opens the requested one.
    """
    spacing = max(config.four_activate_window_ns, config.activate_same_group_ns,
                  config.activate_different_group_ns)
    if config.control_mode == "jedec_hbm":
        # PRE waits tRAS after the previous ACT (a new ACT tRC after it),
        # then tRP; the column command waits tRCD; ACT history from before
        # the window can hold the ACT by up to one tFAW/tRRD window.
        commands = (config.activate_command_cycles + config.precharge_command_cycles + 2.0) * config.command_clock_ns
        return (max(config.minimum_active_ns + config.precharge_to_activate_ns, config.activate_cycle_ns)
                + max(config.activate_to_read_ns, config.activate_to_write_ns)
                + spacing + config.precharge_to_precharge_ns + commands)
    # Analytical banks precharge after their data recovery, then activate.
    recovery = max(config.read_recovery_ns, config.write_recovery_ns)
    return recovery + config.close_ns + config.open_ns + (spacing if config.bank_group_timing_enabled else 0.0)


def _rotated(steps, bank_count, banks_per_group):
    """Vector form of memory_mapping.rotated_bank."""
    unit = (16 if bank_count % 16 == 0 and banks_per_group < 16 and 16 % banks_per_group == 0
            else bank_count)
    groups = unit // banks_per_group if unit % banks_per_group == 0 else 1
    if groups <= 1:
        return steps
    base, offset = np.divmod(steps, unit)
    return base * unit + (offset % groups) * banks_per_group + offset // groups


class _Device:
    """First accesses of one DRAM owner's banks, mirroring map_dram_address."""

    __slots__ = ("config", "penalty", "lanes", "banks", "dies", "per_die", "per_stack", "seen", "unseen")

    def __init__(self, config):
        self.config, self.penalty = config, first_access_penalty_ns(config)
        self.lanes, self.banks = config.lane_count, config.bank_count
        self.dies = config.addressable_die_count
        self.per_die = self.lanes * self.banks * config.rows_per_bank * config.row_bytes
        self.per_stack = self.per_die * self.dies
        total = config.addressable_stack_count * self.dies * self.lanes * self.banks
        self.seen, self.unseen = np.zeros(total, dtype=np.bool_), total

    def _lanes_banks(self, local, count):
        config = self.config
        if config.address_mapping == "row_local":
            first, last = local // config.row_bytes, (local + count - 1) // config.row_bytes
            if last - first + 1 >= self.lanes * self.banks:
                return None
            stripes = np.arange(first, last + 1, dtype=np.int64)
            return (stripes // self.banks) % self.lanes, stripes % self.banks
        interleave = config.interleave_bytes or config.burst_bytes
        first, last = local // interleave, (local + count - 1) // interleave
        if last - first + 1 >= self.lanes * self.banks:
            return None
        blocks = np.arange(first, last + 1, dtype=np.int64)
        return blocks % self.lanes, _rotated((blocks // self.lanes) % self.banks, self.banks,
                                             config.banks_per_group)

    def _paths(self, stack, lanes):
        """Command path of each lane: (stack, channel) for JEDEC HBM, else the lane's command resource."""
        config = self.config
        if config.control_mode != "jedec_hbm":
            return np.zeros_like(lanes) if config.metadata.get("shared_command_resource") else lanes
        per_stack = config.lanes_per_stack
        stacks = lanes // per_stack if config.stack_interface_mode == "independent" else stack
        channels = ((lanes % per_stack) // config.pseudo_channels_per_channel
                    // config.subchannels_per_channel) % config.channels
        return stacks * config.channels + channels

    def first_accesses(self, address, byte_count):
        """Penalty per command path for the banks this access reaches first."""
        end = address + byte_count
        while address < end and self.unseen:
            stop = min(end, (address // self.per_die + 1) * self.per_die)
            stack, die = address // self.per_stack, (address % self.per_stack) // self.per_die
            mapped = self._lanes_banks(address % self.per_die, stop - address)
            if mapped is None:
                lanes = np.repeat(np.arange(self.lanes, dtype=np.int64), self.banks)
                banks = np.tile(np.arange(self.banks, dtype=np.int64), self.lanes)
            else:
                lanes, banks = mapped
            keys = ((stack * self.dies + die) * self.lanes + lanes) * self.banks + banks
            keys, first = np.unique(keys, return_index=True)
            fresh = ~self.seen[keys]
            if fresh.any():
                self.seen[keys[fresh]] = True
                self.unseen -= int(fresh.sum())
                paths, counts = np.unique(self._paths(stack, lanes[first[fresh]]), return_counts=True)
                yield from zip(paths.tolist(), (count * self.penalty for count in counts.tolist()))
            address = stop


def first_access_ns(facts, configs, term_of):
    """Largest boundary row-state delay of each term, in ns.

    ``facts`` lists every executed task as ``(task_id, start_ns, end_ns,
    dependencies, resource_ids, accesses)`` with ``accesses`` a sequence of
    ``(owner, address, byte_count)``. ``configs`` maps owners to their bound
    physical configs (owners without DRAM rows are skipped) and ``term_of``
    maps the task ids of every bounded term to its key.
    """
    devices = {owner: _Device(config) for owner, config in configs.items()
               if isinstance(config, DramConfig)}
    ordered = sorted(range(len(facts)), key=lambda index: (facts[index][1], facts[index][2], index))
    delay, last_user, result = {}, {}, {}
    for index in ordered:
        task_id, _start, _end, dependencies, resources, accesses = facts[index]
        sums = defaultdict(float)
        for owner, address, byte_count in accesses:
            device = devices.get(owner)
            if device is not None and device.unseen and byte_count > 0:
                for path, penalty in device.first_accesses(address, byte_count):
                    sums[owner, path] += penalty
        term = term_of.get(task_id)
        if term is None:
            for resource in resources:
                last_user[resource] = task_id
            continue
        before = [delay[parent] for parent in dependencies if term_of.get(parent) == term]
        for resource in resources:
            previous = last_user.get(resource)
            if previous is not None and term_of.get(previous) == term:
                before.append(delay[previous])
            last_user[resource] = task_id
        delay[task_id] = max(sums.values(), default=0.) + max(before, default=0.)
        result[term] = max(result.get(term, 0.), delay[task_id])
    return result
