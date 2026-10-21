"""Undo resident arrays or only the published HBM objects changed by a task."""
from __future__ import annotations

import copy


class _NumericUndo:
    """Save only channels written after a transaction begins.

    A channel includes both pseudo channels: refresh and maximum-active
    maintenance can update the sibling of the request's addressed PC.  Dense
    requests promote the journal to one full image of the transaction start.
    """

    def __init__(self, packed):
        self.packed = packed
        self.commits = packed._controller_commits
        self.whole = None
        self.channels = {}
        self.touched = None

    def capture(self, request):
        if self.whole is not None:
            return
        packed = self.packed
        selection = packed.request_selection(request)
        if selection is None:
            self.whole = self.restored_arrays()
            self.channels.clear()
            self.touched = None
            return
        if self.touched is None:
            # execute() clears the entire bitmap, including untouched lanes.
            self.touched = packed.arrays[11].copy()
        channel_rows = selection[6]
        channels = (range(*channel_rows.indices(len(packed.channel_keys)))
                    if isinstance(channel_rows, slice) else channel_rows)
        from ._compiled_hbm import _request_array_selection
        for channel in channels:
            channel = int(channel)
            if channel in self.channels:
                continue
            indices = _request_array_selection(
                packed._config.lane_count, packed._config.bank_count,
                2 * channel, 0)
            # A single-channel device uses the full selection even for a
            # sparse address. It is cheaper and covers the same state.
            if indices is None:
                self.whole = self.restored_arrays()
                self.channels.clear()
                self.touched = None
                return
            self.channels[channel] = (indices, tuple(
                None if number == 11 else
                array[index].copy() if isinstance(index, slice) else array[index]
                for number, (array, index) in enumerate(zip(packed.arrays, indices))))

    def _restore_into(self, arrays):
        for indices, saved in self.channels.values():
            for number, (array, index, values) in enumerate(zip(arrays, indices, saved)):
                if number != 11:
                    array[index] = values
        if self.touched is not None:
            arrays[11][...] = self.touched

    def restored_arrays(self):
        arrays = tuple(array.copy() for array in (
            self.packed.arrays if self.whole is None else self.whole))
        if self.whole is None:
            self._restore_into(arrays)
        return arrays

    def restore(self):
        if self.whole is None:
            self._restore_into(self.packed.arrays)
        else:
            self.packed.arrays = tuple(array.copy() for array in self.whole)
        self.packed._controller_commits = self.commits


class HBMTransaction:
    def __init__(self, core):
        self.roots = dict(core.__dict__["_hbm_control_state"])
        self.objects = {}
        self.positions = {}
        self.full = None
        self.numeric = None
        packed = core._hbm_numeric_state
        if packed is not None:
            self.capture_numeric(packed)

    def capture_numeric(self, packed):
        if self.full is None and self.numeric is None:
            self.numeric = _NumericUndo(packed)

    def capture_numeric_request(self, packed, request):
        self.capture_numeric(packed)
        if self.numeric is not None:
            self.numeric.capture(request)

    def remember_bank(self, container, key):
        position = (id(container), key)
        if self.full is not None or position in self.positions:
            return
        previous = container.get(key)
        self.positions[position] = (container, key, previous)
        if previous is not None and id(previous) not in self.objects:
            self.objects[id(previous)] = (previous, "bank", (
                previous.open_row, previous.ready_ns, previous.activation_ready_ns))

    def remember_entry(self, container, key):
        position = (id(container), key)
        if self.full is not None or position in self.positions:
            return
        previous = container.get(key)
        self.positions[position] = (container, key, previous)
        if previous is not None and id(previous) not in self.objects:
            # Numeric commits replace nested PC histories rather than mutate
            # them. Keeping their old references preserves extension aliases.
            self.objects[id(previous)] = (previous, "dict", previous.copy())

    @staticmethod
    def _restore_object(target, kind, saved):
        if kind == "bank":
            target.open_row, target.ready_ns, target.activation_ready_ns = saved
        else:
            target.clear()
            target.update(saved)

    def materialize(self, core):
        """Upgrade before an executor that may mutate arbitrary nested state."""
        if self.full is not None:
            return
        if self.numeric is not None:
            from ._compiled_hbm import _HBMNumericState
            from .dram_core import DramCore
            packed = self.numeric.packed
            arrays = self.numeric.restored_arrays()
            commits = self.numeric.commits
            shadow = DramCore(core.config, capture_details=False)
            shadow.__dict__["_banks"], shadow.__dict__["_hbm_control_state"] = copy.deepcopy((
                core.__dict__["_banks"], core.__dict__["_hbm_control_state"]))
            restored = _HBMNumericState(shadow, packed.bank_ids, packed.pc_keys,
                packed.channel_keys, packed.ids, packed.g, packed.t, packed.run_numeric,
                arrays)
            restored._controller_before = packed._controller_before
            restored._config = packed._config
            if commits:
                restored.commit_controller()
            self.full = (shadow.__dict__["_banks"], shadow.__dict__["_hbm_control_state"])
            self.numeric = None
            return
        memo = {}
        saved = copy.deepcopy((core.__dict__["_banks"], core.__dict__["_hbm_control_state"]), memo)
        # Rewind earlier numeric commits in the private copy, retaining the
        # original graph aliases even if two keys refer to the same row.
        for original, kind, fields in self.objects.values():
            target = memo.get(id(original))
            if target is None:
                target = copy.deepcopy(original, memo)
            self._restore_object(target, kind, copy.deepcopy(fields, memo))
        for container, key, previous in self.positions.values():
            target = memo[id(container)]
            if previous is None:
                target.pop(key, None)
            else:
                target[key] = copy.deepcopy(previous, memo)
        saved[1].clear()
        saved[1].update(copy.deepcopy(self.roots, memo))
        self.full = saved
        self.objects.clear()
        self.positions.clear()

    def rollback(self, core):
        if self.numeric is not None:
            self.numeric.restore()
            core._hbm_numeric_state = self.numeric.packed
            return
        if self.full is not None:
            core._hbm_numeric_state = None
            core.__dict__["_banks"], core.__dict__["_hbm_control_state"] = copy.deepcopy(self.full)
            return
        for target, kind, saved in self.objects.values():
            self._restore_object(target, kind, saved)
        for container, key, previous in self.positions.values():
            if previous is None:
                container.pop(key, None)
            else:
                container[key] = previous
        core.__dict__["_hbm_control_state"].clear()
        core.__dict__["_hbm_control_state"].update(self.roots)
