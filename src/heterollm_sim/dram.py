"""Small parameterized address-aware DRAM timing model.

This is an analytical burst model, not cycle accurate.  It intentionally does
not infer hardware values: every timing and geometry parameter is supplied by
the caller and evidence is required for the profile.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Tuple


def _pos_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonneg_num(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


@dataclass(frozen=True)
class DramProfile:
    channels: int
    banks_per_channel: int
    burst_bytes: int
    row_bytes: int
    t_rcd_ns: float
    t_rp_ns: float
    t_ras_ns: float
    read_latency_ns: float
    write_latency_ns: float
    read_to_write_ns: float
    write_to_read_ns: float
    read_recovery_ns: float
    write_recovery_ns: float
    refresh_interval_ns: float
    refresh_duration_ns: float
    evidence: str
    aggregate_policy: str = "unknown"
    max_bursts_per_access: int = 65536
    # Optional organization fields. Defaults preserve the original DRAM
    # contract; HBM can use stack/pseudo-channel fields without adopting
    # DDR DIMM rank semantics.
    subchannels_per_channel: int = 1
    ranks_per_channel: int = 1
    bank_groups_per_channel: Optional[int] = None
    banks_per_bank_group: Optional[int] = None
    stack_count: int = 1
    hbm_stacks: Optional[int] = None
    pseudo_channels_per_channel: int = 1
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "channels", "banks_per_channel", "burst_bytes", "row_bytes",
            "max_bursts_per_access", "subchannels_per_channel",
            "ranks_per_channel", "stack_count", "pseudo_channels_per_channel",
        ):
            _pos_int(name, getattr(self, name))
        if self.hbm_stacks is not None:
            _pos_int("hbm_stacks", self.hbm_stacks)
            if self.stack_count != 1 and self.stack_count != self.hbm_stacks:
                raise ValueError("stack_count and hbm_stacks disagree")
            object.__setattr__(self, "stack_count", self.hbm_stacks)
        for name in ("bank_groups_per_channel", "banks_per_bank_group"):
            value = getattr(self, name)
            if value is not None:
                _pos_int(name, value)
        if self.bank_groups_per_channel is not None:
            if self.banks_per_channel % self.bank_groups_per_channel:
                raise ValueError("banks_per_channel must divide evenly into bank groups")
            expected = self.banks_per_channel // self.bank_groups_per_channel
            if self.banks_per_bank_group is not None and self.banks_per_bank_group != expected:
                raise ValueError("banks_per_bank_group disagrees with banks_per_channel")
            object.__setattr__(self, "banks_per_bank_group", expected)
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        if self.row_bytes < self.burst_bytes or self.row_bytes % self.burst_bytes:
            raise ValueError("row_bytes must be a multiple of burst_bytes")
        for name in ("t_rcd_ns", "t_rp_ns", "t_ras_ns", "read_latency_ns", "write_latency_ns", "read_to_write_ns", "write_to_read_ns", "read_recovery_ns", "write_recovery_ns", "refresh_interval_ns", "refresh_duration_ns"):
            _nonneg_num(name, getattr(self, name))
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("evidence must be non-empty")
        if self.aggregate_policy not in {"unknown", "cold_contiguous"}:
            raise ValueError("aggregate_policy must be unknown or cold_contiguous")
        interval, duration = self.refresh_interval_ns, self.refresh_duration_ns
        if (interval == 0) != (duration == 0):
            raise ValueError("refresh interval and duration must both be zero or both be enabled")
        if interval and not duration < interval:
            raise ValueError("refresh_duration_ns must be less than refresh_interval_ns")


@dataclass(frozen=True)
class _BankState:
    open_row: Optional[int] = None
    activation_ns: float = 0.0
    bank_ready_ns: float = 0.0


@dataclass(frozen=True)
class _ChannelState:
    bus_ready_ns: float = 0.0
    last_direction: Optional[str] = None
    refresh_epoch: int = 0


@dataclass(frozen=True)
class DramState:
    banks: Tuple[_BankState, ...]
    channels: Tuple[_ChannelState, ...]


def _lane_count(profile: DramProfile) -> int:
    return (
        profile.stack_count
        * profile.channels
        * profile.subchannels_per_channel
        * profile.pseudo_channels_per_channel
    )


def _initial_state(profile: DramProfile) -> DramState:
    lanes = _lane_count(profile)
    return DramState(
        tuple(_BankState() for _ in range(lanes * profile.ranks_per_channel * profile.banks_per_channel)),
        tuple(_ChannelState() for _ in range(lanes)),
    )


def _validate_state(profile: DramProfile, state: DramState) -> None:
    if not isinstance(state, DramState):
        raise TypeError("state must be DramState or None")
    lanes = _lane_count(profile)
    if len(state.banks) != lanes * profile.ranks_per_channel * profile.banks_per_channel or len(state.channels) != lanes:
        raise ValueError("state geometry does not match profile")


def _address(profile: DramProfile, burst_index: int) -> Tuple[int, int, int, int, int, Optional[int], int, int, int]:
    """Map a burst to explicit stack/lane/rank/bank coordinates."""
    per_row = profile.row_bytes // profile.burst_bytes
    lanes = _lane_count(profile)
    lane_linear = burst_index % lanes
    lane_rest = lane_linear
    pseudo = lane_rest % profile.pseudo_channels_per_channel
    lane_rest //= profile.pseudo_channels_per_channel
    subchannel = lane_rest % profile.subchannels_per_channel
    lane_rest //= profile.subchannels_per_channel
    channel = lane_rest % profile.channels
    stack = lane_rest // profile.channels
    bank_linear = burst_index // lanes
    rank = bank_linear % profile.ranks_per_channel
    bank_linear //= profile.ranks_per_channel
    bank = bank_linear % profile.banks_per_channel
    column_linear = bank_linear // profile.banks_per_channel
    column = column_linear % per_row
    row = column_linear // per_row
    bank_group = (
        bank // profile.banks_per_bank_group
        if profile.bank_groups_per_channel is not None and profile.banks_per_bank_group
        else None
    )
    return stack, channel, subchannel, pseudo, rank, bank_group, bank, column, row


def dram_service(
    profile: DramProfile,
    accesses: Tuple[Mapping[str, Any], ...],
    *,
    read_bandwidth_gb_s: float,
    write_bandwidth_gb_s: float,
    max_outstanding_requests: int,
    start_ns: float = 0.0,
    state: Optional[DramState] = None,
) -> Tuple[Mapping[str, Any], DramState]:
    """Preview one owner-serialized access batch and return immutable next state."""
    if not isinstance(profile, DramProfile):
        raise TypeError("profile must be DramProfile")
    _nonneg_num("start_ns", start_ns)
    read_bw, write_bw = _nonneg_num("read_bandwidth_gb_s", read_bandwidth_gb_s), _nonneg_num("write_bandwidth_gb_s", write_bandwidth_gb_s)
    if read_bw <= 0 or write_bw <= 0:
        raise ValueError("read/write bandwidth must be positive")
    _pos_int("max_outstanding_requests", max_outstanding_requests)
    if not isinstance(accesses, tuple):
        raise TypeError("accesses must be a tuple")
    total_bursts = 0
    parsed = []
    for item in accesses:
        if not isinstance(item, Mapping) or item.get("operation") not in {"read", "write"}:
            raise ValueError("access operation must be read or write")
        offset, count = item.get("offset_bytes"), item.get("byte_count")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset_bytes must be a non-negative integer")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("byte_count must be a positive integer")
        first, last = offset // profile.burst_bytes, (offset + count - 1) // profile.burst_bytes
        bursts = last - first + 1
        total_bursts += bursts
        parsed.append((str(item["operation"]), offset, count, first, bursts))
    if total_bursts > profile.max_bursts_per_access:
        raise ValueError("access burst count exceeds max_bursts_per_access")
    previous = _initial_state(profile) if state is None else state
    _validate_state(profile, previous)
    banks = list(previous.banks)
    channels = list(previous.channels)
    metrics = {
        # Keep the established model identity for callers; the additional
        # organization coordinates are additive metadata.
        "model": "dram_addressed_burst_v1",
        "model_version": 2,
        "aggregate_policy": profile.aggregate_policy,
        "evidence": profile.evidence,
        "provenance": dict(profile.provenance),
        "organization_provenance": {
            name: (
                "unknown" if value is None else
                str(profile.provenance.get(name, "parameterized"))
            )
            for name, value in (
                ("subchannels_per_channel", profile.subchannels_per_channel),
                ("ranks_per_channel", profile.ranks_per_channel),
                ("bank_groups_per_channel", profile.bank_groups_per_channel),
                ("banks_per_bank_group", profile.banks_per_bank_group),
                ("stack_count", profile.stack_count),
                ("hbm_stacks", profile.stack_count),
                ("pseudo_channels_per_channel", profile.pseudo_channels_per_channel),
            )
        },
        "lane_count": _lane_count(profile),
        "stack_count": profile.stack_count,
        "channels_per_stack": profile.channels,
        "subchannels_per_channel": profile.subchannels_per_channel,
        "pseudo_channels_per_channel": profile.pseudo_channels_per_channel,
        "ranks_per_channel": profile.ranks_per_channel,
        "bank_groups_per_channel": profile.bank_groups_per_channel,
        "logical_read_bytes": 0,
        "logical_write_bytes": 0,
        "physical_read_bytes": 0,
        "physical_write_bytes": 0,
        "burst_count": total_bursts,
        "row_hits": 0,
        "row_misses": 0,
        "row_conflicts": 0,
        "turnaround_ns": 0.0,
        "refresh_wait_ns": 0.0,
        "queue_wait_ns": 0.0,
        "bank_busy_ns": 0.0,
        "channel_busy_ns": 0.0,
    }
    boundaries = []
    bursts_seen = 0
    finish = float(start_ns)
    for operation, offset, count, first, bursts in parsed:
        metrics["logical_read_bytes" if operation == "read" else "logical_write_bytes"] += count
        physical = bursts * profile.burst_bytes
        metrics["physical_read_bytes" if operation == "read" else "physical_write_bytes"] += physical
        bw = read_bw / _lane_count(profile) if operation == "read" else write_bw / _lane_count(profile)
        burst_data_ns = profile.burst_bytes / bw
        for ordinal in range(bursts):
            index = first + ordinal
            (
                stack_id, channel_id, subchannel_id, pseudo_channel_id,
                rank_id, bank_group_id, bank_id, column, row,
            ) = _address(profile, index)
            lane_id = (
                ((stack_id * profile.channels + channel_id)
                 * profile.subchannels_per_channel + subchannel_id)
                * profile.pseudo_channels_per_channel + pseudo_channel_id
            )
            bank_index = (
                (lane_id * profile.ranks_per_channel + rank_id)
                * profile.banks_per_channel + bank_id
            )
            bank, channel = banks[bank_index], channels[lane_id]
            now = max(float(start_ns), bank.bank_ready_ns, channel.bus_ready_ns)
            if profile.refresh_interval_ns:
                epoch = int(math.floor(now / profile.refresh_interval_ns))
                boundary = (epoch + 1) * profile.refresh_interval_ns
                if now >= boundary or channel.refresh_epoch < epoch:
                    wait = max(0.0, boundary - now) if now < boundary else profile.refresh_duration_ns
                    refresh_end = (boundary if now < boundary else now) + profile.refresh_duration_ns
                    metrics["refresh_wait_ns"] += wait + profile.refresh_duration_ns
                    lane_base = lane_id * profile.ranks_per_channel * profile.banks_per_channel
                    for b in range(profile.ranks_per_channel * profile.banks_per_channel):
                        banks[lane_base + b] = _BankState()
                    bank = banks[bank_index]
                    channel = _ChannelState(refresh_end, None, epoch + 1)
                    channels[lane_id] = channel
                    now = refresh_end
            command_ready = max(now, bank.bank_ready_ns)
            if bank.open_row == row:
                metrics["row_hits"] += 1
                activate = command_ready
            else:
                if bank.open_row is None:
                    metrics["row_misses"] += 1
                    activate = command_ready
                else:
                    metrics["row_conflicts"] += 1
                    activate = max(command_ready, bank.activation_ns + profile.t_ras_ns) + profile.t_rp_ns
                activate += profile.t_rcd_ns
                bank = _BankState(row, activate - profile.t_rcd_ns, activate)
            queue_gate = float(start_ns) + (bursts_seen // max_outstanding_requests) * burst_data_ns
            data_start = max(activate if bank.open_row == row else command_ready, channel.bus_ready_ns, queue_gate)
            if channel.last_direction and channel.last_direction != operation:
                switch = profile.read_to_write_ns if channel.last_direction == "read" else profile.write_to_read_ns
                data_start += switch
                metrics["turnaround_ns"] += switch
            data_end = data_start + (profile.read_latency_ns if operation == "read" else profile.write_latency_ns) + burst_data_ns
            recovery = profile.read_recovery_ns if operation == "read" else profile.write_recovery_ns
            banks[bank_index] = _BankState(bank.open_row, bank.activation_ns, data_end + recovery)
            channels[lane_id] = _ChannelState(data_end, operation, channel.refresh_epoch)
            metrics["queue_wait_ns"] += max(0.0, data_start - max(activate if bank.open_row == row else command_ready, channel.bus_ready_ns))
            metrics["bank_busy_ns"] += data_end - data_start + recovery
            metrics["channel_busy_ns"] += data_end - max(data_start, channel.bus_ready_ns)
            finish = max(finish, data_end)
            if len(boundaries) < 128:
                boundaries.append({
                    "burst_index": index,
                    "stack": stack_id,
                    "channel": channel_id,
                    "subchannel": subchannel_id,
                    "pseudo_channel": pseudo_channel_id,
                    "rank": rank_id,
                    "bank_group": bank_group_id,
                    "bank": bank_id,
                    "column": column,
                    "row": row,
                    "operation": operation,
                    "start_ns": data_start,
                    "end_ns": data_end,
                })
            bursts_seen += 1
    metrics["physical_bytes"] = metrics["physical_read_bytes"] + metrics["physical_write_bytes"]
    metrics["service_ns"] = max(0.0, finish - float(start_ns))
    metrics["boundaries"] = tuple(boundaries)
    return metrics, DramState(tuple(banks), tuple(channels))


def resolve_dram_task(
    task: Any,
    states: Mapping[str, DramState],
    *,
    start_ns: Optional[float] = None,
):
    """Preview an address-aware DRAM task without mutating kernel state."""

    contract = task.metadata.get("dram_access")
    if not isinstance(contract, Mapping):
        return task, None, None
    profile = contract.get("profile")
    if isinstance(profile, Mapping):
        profile = DramProfile(**dict(profile))
    if not isinstance(profile, DramProfile):
        raise TypeError("dram_access.profile must be a DramProfile or mapping")
    accesses = contract.get("accesses")
    if not isinstance(accesses, (tuple, list)):
        raise ValueError("dram_access.accesses must be a sequence")
    state_key = str(contract.get("state_key") or profile.evidence)
    previous = states.get(state_key)
    metrics, next_state = dram_service(
        profile,
        tuple(accesses),
        read_bandwidth_gb_s=contract["read_bandwidth_gb_s"],
        write_bandwidth_gb_s=contract["write_bandwidth_gb_s"],
        max_outstanding_requests=contract["max_outstanding_requests"],
        start_ns=(
            float(contract.get("start_ns", 0.0))
            if start_ns is None else float(start_ns)
        ),
        state=previous,
    )
    resource_id = str(contract.get("resource_id") or "")
    if not resource_id:
        raise ValueError("dram_access.resource_id must be non-empty")
    if all(demand.resource_id != resource_id for demand in task.demands):
        raise ValueError("dram_access.resource_id must name a task demand")
    resolved = replace(
        task,
        demands=tuple(
            replace(
                demand,
                service_ns=float(metrics["service_ns"]),
                bytes_moved=int(metrics["physical_bytes"]),
            )
            if demand.resource_id == resource_id else demand
            for demand in task.demands
        ),
        metadata={
            **task.metadata,
            "dram_execution": metrics,
            "dram_address_scope": "explicit",
            "dram_timing_completeness": "parameterized_burst_envelope",
        },
    )
    return resolved, state_key, next_state


__all__ = ["DramProfile", "DramState", "dram_service", "resolve_dram_task"]
