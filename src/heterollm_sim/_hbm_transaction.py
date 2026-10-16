"""Undo resident arrays or only the published HBM objects changed by a task."""
from __future__ import annotations

import copy


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
            self.numeric = (packed, tuple(array.copy() for array in packed.arrays), packed._controller_commits)

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
            packed, arrays, commits = self.numeric
            shadow = DramCore(core.config, capture_details=False)
            shadow.__dict__["_banks"], shadow.__dict__["_hbm_control_state"] = copy.deepcopy((
                core.__dict__["_banks"], core.__dict__["_hbm_control_state"]))
            restored = _HBMNumericState(shadow, packed.bank_ids, packed.pc_keys,
                packed.channel_keys, packed.ids, packed.g, packed.t, packed.run_numeric,
                tuple(array.copy() for array in arrays))
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
            packed, arrays, commits = self.numeric
            packed.arrays = tuple(array.copy() for array in arrays)
            packed._controller_commits = commits
            core._hbm_numeric_state = packed
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
