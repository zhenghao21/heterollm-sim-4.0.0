"""Ordered HBM normal-operation commands with explicit JEDEC timing inputs.

The supported subset is ACT/PREpb/PREab/RD/WR/REFab in pseudo-channel mode.
Command occupancy, array eligibility and DQ transfer are separate quantities.
Configured maximum row-active deadlines also apply with refresh disabled.
Training, auto-precharge and refresh postponement are not
implemented. Timing values are supplied by DramConfig, with their provenance
owned by the configuration builder rather than inferred here.
"""
from __future__ import annotations

import math
from bisect import insort

from .memory_mapping import split_dram_request
from .memory_types import Operation, StageTiming, TransactionResult
from ._hbm_clock import align_clock, clock_after


class _Controller:
    def __init__(self, core, request):
        self.core, self.config, self.timeline, self.request = core, core.config, core.timeline, request
        self.state = core._hbm_control_state
        self.banks = self.state.setdefault("banks", {})
        self.pcs = self.state.setdefault("pcs", {})
        self.channels = self.state.setdefault("channels", {})
        self._active_deadlines = {}
        self.clock = self.config.command_clock_ns
        self.hbm2 = self.config.generation in {"HBM2", "HBM2E"}
        self.row_step = self.clock if self.hbm2 else self.clock / 2
        self.validate_calendars()
        self.stages, self.mappings = [], []
        self.truncated = False
        self.counts = dict(activate_commands=0, precharge_commands=0,
                           read_commands=0, write_commands=0,
                           row_command_bus_busy_ns=0.0, column_command_bus_busy_ns=0.0)
        if self.config.refresh_policy == "all_bank":
            self.counts["refresh_commands"] = 0

    def validate_calendars(self):
        ids = {self.bus_id((stack, channel), kind)
               for stack in range(self.config.stacks)
               for channel in range(self.config.channels) for kind in ("row", "column")}
        prefix = self.config.metadata.get("data_resource_prefix", "dram:data")
        ids.update(self.timeline.canonical_resource_id(f"{prefix}:{lane}")
                   for lane in range(self.config.lane_count))
        if any(len(self.timeline.lane_available[rid]) != 1
               for rid in ids if rid in self.timeline.lane_available):
            raise ValueError("JEDEC HBM command and data buses require one execution slot")

    @staticmethod
    def _pc_key(mapping):
        return (mapping.stack, mapping.channel, mapping.pseudo_channel)

    def pc(self, key):
        return self.pcs.setdefault(key, {"activate": {}, "activate_window": [],
            "column": {"read": {}, "write": {}}, "write_end": {},
            "last_precharge": None, "blocked_until": 0.0,
            "next_data_start": {"read": 0.0, "write": 0.0}})

    def bank(self, mapping):
        from .dram_core import _BankState
        rid = self.core._bank_id(mapping)
        old = self.core._banks.setdefault(rid, _BankState())
        state = self.banks.setdefault(rid, {"mapping": mapping, "last_activate": None,
            "activate_ready": 0.0, "precharge_ready": 0.0,
            "read_ready": 0.0, "write_ready": 0.0})
        return old, state

    def record(self, name, start, end, resource=None, byte_count=0):
        if self.core.capture_details and not self.truncated:
            if len(self.stages) >= 6 * self.config.max_expanded_segments:
                self.stages.clear(); self.mappings.clear(); self.truncated = True
                return
            self.stages.append(StageTiming(name, start, end, resource, byte_count))

    def bus_id(self, key, kind):
        prefix = self.config.metadata.get(kind + "_command_resource_prefix", "dram:" + kind + "_command")
        return self.timeline.canonical_resource_id(f"{prefix}:{key[0]}:{key[1]}")

    align = staticmethod(align_clock)

    def after(self, value, delay):
        return clock_after(value, delay, self.clock / 2)

    def reserve_clock(self, rid, start, duration, *, direction=None, byte_count=0):
        """Keep clock-defined endpoints canonical in the shared calendar."""
        start = max(start, self.timeline.available_ns(rid))
        end = self.after(start, duration)
        previous_busy = self.timeline.busy_ns.get(rid, 0.0)
        reservation = self.timeline.reserve(rid, start, end - start, direction=direction)
        # Occupancy is the configured duration, independent of subtracting
        # two large represented endpoints to communicate with the calendar.
        self.timeline.busy_ns[rid] = previous_busy + duration
        if byte_count:
            self.timeline.bytes_moved[rid] = self.timeline.bytes_moved.get(rid, 0) + byte_count
            if (self.timeline._intervals.get(rid)
                    and self.timeline._intervals[rid][-1] == (reservation.start_ns, reservation.end_ns)):
                self.timeline._interval_payloads.setdefault(rid, []).append(
                    (reservation.start_ns, reservation.end_ns, byte_count))
        return reservation

    def command(self, key, kind, earliest, cycles, name, *, rising=False):
        rid = self.bus_id(key, kind)
        slots = self.timeline.lane_available.get(rid)
        if slots is not None and len(slots) != 1:
            raise ValueError("JEDEC HBM command buses require one execution slot")
        quantum = self.clock if rising or kind == "column" else self.row_step
        start = self.align(max(earliest, self.timeline.available_ns(rid)), quantum)
        transfer = self.reserve_clock(rid, start, cycles * self.clock)
        self.counts[kind + "_command_bus_busy_ns"] += transfer.duration_ns
        self.record(name, transfer.start_ns, transfer.end_ns, transfer.resource_id)
        return transfer

    def precharge_ready(self, key, earliest):
        previous = self.pc(key)["last_precharge"]
        return max(earliest, self.after(previous, self.config.precharge_to_precharge_ns)) if previous is not None else earliest

    def precharge(self, mapping, old, bank, earliest):
        key = self._pc_key(mapping)
        ready = self.precharge_ready(key, max(earliest, bank["precharge_ready"], self.pc(key)["blocked_until"]))
        channel = self.channels.get(key[:2], {})
        if channel.get("last_activate_bank") == self.core._bank_id(mapping):
            ready = max(ready, self.after(channel["last_activate_start"], 2 * self.clock))
        command = self.command(key, "row", ready, self.config.precharge_command_cycles, "PRECHARGE")
        self.counts["precharge_commands"] += 1
        self.pc(key)["last_precharge"] = command.start_ns
        old.open_row = None
        self._active_deadlines.pop(key[:2], None)
        bank["activate_ready"] = self.after(command.start_ns, self.config.precharge_to_activate_ns)
        old.ready_ns = bank["activate_ready"]
        self.record("PRECHARGE_RECOVERY", command.start_ns, bank["activate_ready"])

    def activate_start(self, mapping, bank, earliest):
        config, key = self.config, self._pc_key(mapping)
        pc = self.pc(key)
        ready = max(earliest, bank["activate_ready"], pc["blocked_until"],
                    self.timeline.available_ns(self.bus_id(key, "row")))
        if bank["last_activate"] is not None:
            ready = max(ready, self.after(self.after(bank["last_activate"], config.activate_cycle_ns), -self.clock))
        # ACT array timings are anchored at the second rising CK edge.
        # Align the external/request constraint before shifting its anchor;
        # adding and subtracting a CK must never erase a late arrival's ULP.
        anchor = self.after(self.align(ready, self.clock), self.clock)
        # Unlike an arbitrary bank-ready time, the shared row-port frontier
        # proves that no later HBM ACT can be inserted before it. Pruning at
        # this frontier bounds the live tFAW calendar even inside a huge request.
        frontier = max(self.request.arrival_ns, self.timeline.available_ns(self.bus_id(key, "row")))
        while True:
            anchor = self.core._bank_group_activate_ready(pc, mapping.bank_group, anchor,
                                                          frontier, clock_quantum=self.clock / 2)
            start = self.align(self.after(anchor, -self.clock), self.clock)
            aligned = self.after(start, self.clock)
            # Rounding a candidate can enter a window of previously reserved
            # future ACTs. Recheck the aligned anchor before committing it.
            if config.four_activate_window_ns <= 0 or aligned == anchor:
                return start
            anchor = aligned

    def activate(self, mapping, old, bank, row, earliest):
        key = self._pc_key(mapping)
        start = self.activate_start(mapping, bank, earliest)
        command = self.command(key, "row", start, self.config.activate_command_cycles, "ACTIVATE", rising=True)
        anchor = self.after(command.start_ns, self.clock)
        pc = self.pc(key)
        pc["activate"][mapping.bank_group] = anchor
        if self.config.four_activate_window_ns > 0:
            insort(pc["activate_window"], anchor)
        self.channels.setdefault(key[:2], {}).update(last_activate_start=command.start_ns,
            last_activate_bank=self.core._bank_id(mapping), last_activate_pc=key)
        bank["last_activate"] = anchor
        if key[:2] in self._active_deadlines:
            self._active_deadlines[key[:2]] = min(self._active_deadlines[key[:2]],
                                                 self.after(anchor, self.config.maximum_active_ns))
        bank["read_ready"] = self.after(anchor, self.config.activate_to_read_ns)
        bank["write_ready"] = self.after(anchor, self.config.activate_to_write_ns)
        bank["precharge_ready"] = self.after(anchor, self.config.minimum_active_ns)
        old.open_row = row
        old.activation_ready_ns = min(bank["read_ready"], bank["write_ready"])
        old.ready_ns = bank["precharge_ready"]
        self.counts["activate_commands"] += 1

    def column_start(self, mapping, bank_ready, earliest):
        config, key, op = self.config, self._pc_key(mapping), self.request.operation
        pc = self.pc(key)
        sid = (mapping.bank_group * config.banks_per_group + mapping.bank) // 16
        ready = max(earliest, bank_ready, pc["blocked_until"],
                    self.timeline.available_ns(self.bus_id(key, "column")))
        for (previous_sid, group), previous in pc["column"][op.value].items():
            gap = (config.column_same_group_ns if group == mapping.bank_group
                   else config.column_different_group_ns)
            if op is Operation.READ and previous_sid != sid:
                gap = config.column_different_sid_ns
            ready = max(ready, self.after(previous, gap))
        if op is Operation.READ:
            for (_sid, group), end in pc["write_end"].items():
                gap = (config.write_to_read_same_group_ns if group == mapping.bank_group
                       else config.write_to_read_different_group_ns)
                ready = max(ready, self.after(end, gap))
        else:
            for previous in pc["column"]["read"].values():
                ready = max(ready, self.after(previous, config.read_to_write_command_ns))
        latency = config.read_latency_ns if op is Operation.READ else config.write_latency_ns
        data_id = self.core._data_id(mapping)
        slots = self.timeline.lane_available.get(data_id)
        if slots is not None and len(slots) != 1:
            raise ValueError("JEDEC HBM pseudo-channel data buses require one execution slot")
        data_ready = self.timeline.available_ns(data_id)
        previous_direction = self.timeline.directions.get(data_id)
        if previous_direction and previous_direction != op.value:
            data_ready = self.after(data_ready, config.write_to_read_ns if op is Operation.READ else config.read_to_write_ns)
        data_ready = max(data_ready, pc["next_data_start"][op.value])
        # Issue only when the fixed RL/WL pipeline can meet its DQ slot.
        start = self.align(max(ready, self.after(data_ready, -latency)), self.clock)
        # Subtraction from an externally reserved DQ time can lose a ULP.
        # Check the actual fixed-latency endpoint before accepting the edge.
        if self.after(start, latency) < data_ready:
            start = self.after(start, self.clock)
        return start

    def estimate_access(self, mapping, old, bank, row, activate_not_before):
        earliest = self.request.arrival_ns
        if old.open_row != row:
            if old.open_row is not None:
                key = self._pc_key(mapping)
                pre = self.align(max(self.precharge_ready(key, max(earliest, bank["precharge_ready"])),
                                     self.timeline.available_ns(self.bus_id(key, "row"))), self.row_step)
                channel = self.channels.get(key[:2], {})
                if channel.get("last_activate_bank") == self.core._bank_id(mapping):
                    pre = max(pre, self.after(channel["last_activate_start"], 2 * self.clock))
                earliest = self.after(pre, max(self.config.precharge_to_activate_ns,
                                     self.config.precharge_command_cycles * self.clock))
            anchor = self.after(self.activate_start(mapping, bank, max(earliest, activate_not_before)), self.clock)
            bank_ready = self.after(anchor, self.config.activate_to_read_ns if self.request.operation is Operation.READ
                                   else self.config.activate_to_write_ns)
        else:
            anchor = bank["last_activate"]
            bank_ready = bank["read_ready" if self.request.operation is Operation.READ else "write_ready"]
        return self.column_start(mapping, bank_ready, earliest), anchor

    def maximum_active_pending(self, mapping, horizon):
        maximum = self.config.maximum_active_ns
        if maximum <= 0:
            return []
        channel = self._pc_key(mapping)[:2]
        if channel not in self._active_deadlines:
            self._active_deadlines[channel] = min((self.after(bank["last_activate"], maximum)
                for rid, bank in self.banks.items()
                if self.core._banks[rid].open_row is not None
                and self._pc_key(bank["mapping"])[:2] == channel), default=math.inf)
        if self._active_deadlines[channel] > horizon:
            return []
        pending = [(self.after(bank["last_activate"], maximum), rid, bank)
                   for rid, bank in self.banks.items()
                   if self.core._banks[rid].open_row is not None
                   and self._pc_key(bank["mapping"])[:2] == channel
                   and self.after(bank["last_activate"], maximum) <= horizon]
        return sorted(pending, key=lambda item: item[0])

    def close_expiring_rows(self, pending):
        """Place PREs before their tRASmax deadlines on the shared row bus.

        Plan backwards so adjacent deadlines and same-PC tPPD leave room
        for every required PRE. Validate the whole plan before reserving it.
        """
        duration = self.config.precharge_command_cycles * self.clock
        next_start, next_pc, planned = math.inf, {}, []
        for deadline, rid, bank in reversed(pending):
            mapping = bank["mapping"]
            key = self._pc_key(mapping)
            latest = min(deadline, self.after(next_start, -duration),
                         self.after(next_pc.get(key, math.inf), -self.config.precharge_to_precharge_ns))
            start = math.floor(latest / self.row_step) * self.row_step
            # Use the same upward-alignment arithmetic as command() even
            # for operating points whose CK is not exactly binary-float.
            while self.align(start, self.row_step) > latest:
                start = self.after(start, -self.row_step)
            start = self.align(start, self.row_step)
            earliest = self.precharge_ready(key, max(bank["precharge_ready"],
                self.pc(key)["blocked_until"], self.timeline.available_ns(self.bus_id(key, "row"))))
            channel = self.channels.get(key[:2], {})
            if channel.get("last_activate_bank") == rid:
                earliest = max(earliest, self.after(channel["last_activate_start"], 2 * self.clock))
            if start < earliest:
                raise ValueError("HBM tRASmax deadline cannot be met by a legal PRE command")
            planned.append((mapping, rid, bank, start))
            next_start, next_pc[key] = start, start
        for mapping, rid, bank, start in reversed(planned):
            self.precharge(mapping, self.core._banks[rid], bank, start)

    def next_refresh_due(self, mapping):
        key = self._pc_key(mapping)
        channel = self.channels.setdefault(key[:2], {})
        interval = self.config.refresh_interval_ns
        if "next_refresh" not in channel:
            # Model the current maintenance epoch, not pre-model history.
            channel["next_refresh"] = max(interval, math.floor(self.request.arrival_ns / interval) * interval)
        return channel["next_refresh"]

    def refresh_pc(self, key, due):
        pc = self.pc(key)
        selected = [(self.core._banks[rid], bank) for rid, bank in self.banks.items()
                    if self._pc_key(bank["mapping"]) == key]
        opened = [(old, bank) for old, bank in selected if old.open_row is not None]
        earliest = max(due, pc["blocked_until"])
        if opened:
            earliest = self.precharge_ready(key, max(earliest, *(bank["precharge_ready"] for _, bank in opened)))
            channel = self.channels.get(key[:2], {})
            if channel.get("last_activate_pc") == key:
                earliest = max(earliest, self.after(channel["last_activate_start"], 2 * self.clock))
            pre_start = self.align(max(earliest, self.timeline.available_ns(self.bus_id(key, "row"))),
                                   self.row_step)
            if (self.config.maximum_active_ns > 0 and any(
                    pre_start > self.after(bank["last_activate"], self.config.maximum_active_ns)
                    for _, bank in opened)):
                raise ValueError("HBM tRASmax deadline cannot be met by the refresh PRE command")
            pre = self.command(key, "row", earliest, self.config.precharge_command_cycles, "PRECHARGE_ALL")
            self.counts["precharge_commands"] += 1
            pc["last_precharge"] = pre.start_ns
            earliest = self.after(pre.start_ns, self.config.precharge_to_activate_ns)
            for old, bank in opened:
                old.open_row = None
                bank["activate_ready"] = earliest
                old.ready_ns = earliest
        earliest = max(earliest, *(bank["activate_ready"] for _, bank in selected)) if selected else earliest
        last_act = self.channels.get(key[:2], {}).get("last_activate_start")
        if last_act is not None:
            # The final half of ACT's second CK only admits restricted PRE.
            earliest = max(earliest, self.after(last_act, 2 * self.clock))
        if self.hbm2:
            earliest = max(earliest, self.timeline.available_ns(self.bus_id(key, "column")))
        ref = self.command(key, "row", earliest, self.config.precharge_command_cycles, "REFRESH_ALL", rising=True)
        if self.hbm2:
            self.command(key, "column", ref.start_ns, 1.0, "REFRESH_CNOP")
        if ref.start_ns > due + 8 * self.config.refresh_interval_ns:
            raise ValueError("HBM periodic refresh missed the maximum eight postponed intervals")
        pc["blocked_until"] = self.after(ref.start_ns, self.config.refresh_cycle_ns)
        self._active_deadlines.pop(key[:2], None)
        for old, bank in selected:
            old.open_row = None
            bank["activate_ready"] = pc["blocked_until"]
            old.ready_ns = pc["blocked_until"]
        self.counts["refresh_commands"] += 1
        self.record("REFRESH_RECOVERY", ref.start_ns, pc["blocked_until"])

    def refresh_until(self, mapping, horizon):
        if self.config.refresh_policy != "all_bank":
            return False
        key = self._pc_key(mapping)
        channel = self.channels.setdefault(key[:2], {})
        interval = self.config.refresh_interval_ns
        self.next_refresh_due(mapping)
        refreshed = False
        while channel["next_refresh"] <= horizon:
            if self.fast_forward_idle_refresh(key, channel):
                refreshed = True
                continue
            due = channel["next_refresh"]
            for pseudo_channel in range(self.config.pseudo_channels_per_channel):
                self.refresh_pc((*key[:2], pseudo_channel), due)
            channel["next_refresh"] = self.after(channel["next_refresh"], interval)
            refreshed = True
        return refreshed

    def fast_forward_idle_refresh(self, key, channel):
        """Aggregate only closed, fully idle maintenance before this arrival.

        Each skipped epoch consists of exactly one REFab per PC. No normal
        command, open row, timing gate or overlapping refresh is skipped.
        """
        if self.timeline.resource_aliases:
            return False
        config = self.config
        interval, due = config.refresh_interval_ns, channel["next_refresh"]
        # Preserve the exact serial float64 busy/counter values. With a
        # half-ns clock grid, command endpoints/occupancy are quarter-ns
        # values. Outside its exact range, retain individual reservations.
        grid_values = (self.clock, interval, due, self.request.arrival_ns,
                       config.refresh_cycle_ns)
        if any(value >= 2 ** 49 or not float(value * 2).is_integer() for value in grid_values):
            return False
        pc_count = config.pseudo_channels_per_channel
        if interval < config.refresh_cycle_ns + pc_count * self.clock:
            return False
        cutoff = self.request.arrival_ns - config.refresh_cycle_ns - pc_count * self.clock
        periods = math.floor((cutoff - due) / interval) + 1
        if periods < 2 or (self.core.capture_details and periods * pc_count <= config.max_expanded_segments):
            return False
        selected = [(rid, bank) for rid, bank in self.banks.items()
                    if self._pc_key(bank["mapping"])[:2] == key[:2]]
        if any(self.core._banks[rid].open_row is not None or bank["activate_ready"] > due
               for rid, bank in selected):
            return False
        row_id, column_id = self.bus_id(key, "row"), self.bus_id(key, "column")
        if any(not float(self.timeline.busy_ns.get(rid, 0.0) * 4).is_integer()
               for rid in (row_id, column_id)):
            return False
        if max(self.timeline.available_ns(row_id), self.timeline.available_ns(column_id)) > due:
            return False
        if any(self.pc((*key[:2], pc))["blocked_until"] > due for pc in range(pc_count)):
            return False
        from .dram_core import _repeat_add
        last_due = _repeat_add(due, interval, periods - 1)
        last_start = self.align(last_due, self.clock)
        commands = periods * pc_count
        duration = config.precharge_command_cycles * self.clock
        resources = [("row", row_id, duration)]
        if self.hbm2:
            resources.append(("column", column_id, self.clock))
        for kind, rid, occupancy in resources:
            prior = self.timeline.busy_ns.get(rid, 0.0)
            self.timeline.reserve(rid, last_start + (pc_count - 1) * self.clock, occupancy)
            self.timeline.busy_ns[rid] = _repeat_add(prior, occupancy, commands)
            self.counts[kind + "_command_bus_busy_ns"] += commands * occupancy
        for pseudo in range(pc_count):
            pc = self.pc((*key[:2], pseudo))
            pc["blocked_until"] = last_start + pseudo * self.clock + config.refresh_cycle_ns
        for rid, bank in selected:
            until = self.pc(self._pc_key(bank["mapping"]))["blocked_until"]
            bank["activate_ready"] = until
            self.core._banks[rid].ready_ns = until
        channel["next_refresh"] = _repeat_add(due, interval, periods)
        self.counts["refresh_commands"] += commands
        self.timeline._intervals_truncated = True
        if self.core.capture_details:
            self.stages.clear(); self.mappings.clear(); self.truncated = True
        return True

    def execute(self):
        config, request = self.config, self.request
        completion = request.arrival_ns
        logical = physical = hits = misses = conflicts = count = 0
        for segment in split_dram_request(request, config):
            count += 1
            if count > config.max_expanded_segments and not self.truncated:
                self.stages.clear(); self.mappings.clear(); self.truncated = True
            mapping = segment.mapping
            if self.core.capture_details and not self.truncated:
                self.mappings.append(mapping)
            old, bank = self.bank(mapping)
            row = mapping.row if config.activated_row_msb is None else mapping.row & ~(1 << config.activated_row_msb)
            activate_not_before = request.arrival_ns
            # Check the command's recovery horizon before issue, including
            # every open bank sharing this physical channel's row CA bus.
            for _ in range(2 * config.bank_count + 16):
                command_start, anchor = self.estimate_access(mapping, old, bank, row, activate_not_before)
                recovery = (config.read_to_precharge_ns if request.operation is Operation.READ else
                            self.after(self.after(config.write_latency_ns,
                                segment.transfer_bytes / config.effective_lane_bandwidth_gb_s),
                                config.write_to_precharge_ns))
                horizon = self.after(command_start, recovery)
                pending = self.maximum_active_pending(mapping, horizon)
                if pending:
                    if config.refresh_policy == "all_bank" and self.next_refresh_due(mapping) <= pending[0][0]:
                        self.refresh_until(mapping, self.next_refresh_due(mapping))
                    else:
                        self.close_expiring_rows(pending)
                    continue
                if self.refresh_until(mapping, command_start):
                    continue
                if (config.maximum_active_ns > 0 and old.open_row != row
                        and horizon > self.after(anchor, config.maximum_active_ns)):
                    rcd = config.activate_to_read_ns if request.operation is Operation.READ else config.activate_to_write_ns
                    if self.after(rcd, recovery) > config.maximum_active_ns:
                        raise ValueError("HBM access recovery exceeds the tRASmax activation window")
                    # A fixed DQ reservation or direction cap can postpone
                    # the column command. Keep the row closed during that
                    # wait instead of opening it for longer than tRASmax.
                    activate_not_before = self.align(max(self.after(activate_not_before, self.clock),
                        self.after(self.after(horizon, -config.maximum_active_ns), -self.clock)), self.clock)
                    continue
                break
            else:
                raise ValueError("HBM timing inputs leave no column-command window between maintenance commands")
            if old.open_row == row:
                hits += 1
            else:
                if old.open_row is None:
                    misses += 1
                else:
                    conflicts += 1
                    self.precharge(mapping, old, bank, request.arrival_ns)
                self.activate(mapping, old, bank, row, activate_not_before)
            read = request.operation is Operation.READ
            command_start = self.column_start(mapping, bank["read_ready" if read else "write_ready"], request.arrival_ns)
            key = self._pc_key(mapping)
            command = self.command(key, "column", command_start, config.column_command_cycles,
                                   "READ_COMMAND" if read else "WRITE_COMMAND", rising=True)
            self.counts["read_commands" if read else "write_commands"] += 1
            sid = (mapping.bank_group * config.banks_per_group + mapping.bank) // 16
            self.pc(key)["column"][request.operation.value][(sid, mapping.bank_group)] = command.start_ns
            latency = config.read_latency_ns if read else config.write_latency_ns
            data_ready = self.after(command.start_ns, latency)
            self.record("READ_PIPELINE" if read else "WRITE_PIPELINE", command.start_ns, data_ready)
            data = self.reserve_clock(self.core._data_id(mapping), data_ready,
                segment.transfer_bytes / config.effective_lane_bandwidth_gb_s,
                direction=request.operation.value, byte_count=segment.transfer_bytes)
            self.pc(key)["next_data_start"][request.operation.value] = (
                self.after(data.start_ns, segment.transfer_bytes / config.directional_lane_bandwidth_gb_s(request.operation)))
            self.record("BURST_TRANSFER", data.start_ns, data.end_ns, data.resource_id, segment.transfer_bytes)
            if read:
                bank["precharge_ready"] = max(bank["precharge_ready"], self.after(command.start_ns, config.read_to_precharge_ns))
            else:
                bank["precharge_ready"] = max(bank["precharge_ready"], self.after(data.end_ns, config.write_to_precharge_ns))
                self.pc(key)["write_end"][(sid, mapping.bank_group)] = data.end_ns
            old.ready_ns = bank["precharge_ready"]
            completion = max(completion, data.end_ns)
            logical += segment.logical_bytes
            physical += segment.transfer_bytes
        return TransactionResult(request.request_id, request.operation, request.arrival_ns, completion,
            logical, physical, tuple(self.mappings), tuple(self.stages), {
                **self.counts, "control_mode": "jedec_hbm", "row_hits": hits, "row_misses": misses,
                "row_conflicts": conflicts, "burst_count": count, "details_truncated": self.truncated,
                "physical_read_bytes": physical if request.operation is Operation.READ else 0,
                "physical_write_bytes": physical if request.operation is Operation.WRITE else 0,
                "queue_wait_ns": 0.0, "bandwidth_ceiling_gb_s": config.directional_bandwidth_gb_s(request.operation),
                "host_transfer_bytes": 0, "internal_transfer_bytes": 0,
                "pages_read": 0, "pages_programmed": 0, "erase_operations": 0,
            })


def execute_hbm(core, request):
    c = core.config
    count = (request.address + request.byte_count - 1) // c.burst_bytes - request.address // c.burst_bytes + 1
    # Single-request numeric packing scans every physical bank. For a few
    # bursts on a large device, the same reference controller costs less.
    # Bound Python work to 16 bursts and require 256 packed banks per burst;
    # batches still pack once and large requests still use the numeric loop.
    tiny_request = count <= min(16, c.lane_count * c.bank_count // 256)
    if not core.capture_details and not tiny_request:
        try:
            from ._compiled_hbm import execute_hbm_numeric
        except (ImportError, OSError):
            pass
        else:
            result = execute_hbm_numeric(core, request)
            if result is not None:
                return result
    return _Controller(core, request).execute()
