"""The native NAND controller's state and immutable geometry plans."""
from __future__ import annotations

import numpy as np

from .memory_mapping import map_nand_address


_GEOMETRY_PLANS = {}


def _geometry_plan(core, operation, banks, units, per_channel):
    """Cache immutable address/resource descriptions, never live calendars."""
    config = core.config
    prefixes = tuple(str(config.metadata.get(name, default)) for name, default in (
        ("array_resource_prefix", "nand:array"), ("parallel_resource_prefix", "nand:parallel"),
        ("channel_resource_prefix", "nand:channel"), ("host_resource_id", "nand:host")))
    key = (config.channels, config.targets_per_channel, config.dies_per_target, config.luns_per_die,
           config.planes_per_lun, config.planes_independent, config.parallel_units,
           config.host_channels, config.host_full_duplex, prefixes,
           tuple(core.timeline.resource_aliases.items()), operation)
    plan = _GEOMETRY_PLANS.get(key)
    if plan is not None:
        return plan
    mappings = tuple(map_nand_address(config, bank * 4096) for bank in range(banks))
    bank_keys = tuple(core._bank_key(mapping) for mapping in mappings)
    tag_prefixes = tuple(core._staged_key(mapping)[:-2] for mapping in mappings)
    array_names = tuple(core._array_id(mappings[unit]) for unit in range(units))
    array_resources = tuple(core._array_resource_id(mappings[unit]) for unit in range(units))
    channel_names = tuple(core._channel_id(mappings[channel * per_channel]) for channel in range(config.channels))
    host_names = tuple(core._host_id(mappings[channel * per_channel], operation)
                       for channel in range(config.host_channels))
    nand_resources = tuple(dict.fromkeys((*array_resources, *channel_names, *host_names)))
    lookup = {name: index for index, name in enumerate(nand_resources)}
    ids = tuple(np.array([lookup[name] for name in names], dtype=np.int64)
                for names in (array_resources, channel_names, host_names))
    for values in ids:
        values.flags.writeable = False
    plan = (bank_keys, tag_prefixes, array_names, array_resources, channel_names,
            host_names, nand_resources, *ids)
    if len(_GEOMETRY_PLANS) >= 64:
        _GEOMETRY_PLANS.clear()
    _GEOMETRY_PLANS[key] = plan
    return plan


class _NandNumericState:
    """Exclusive native controller state; fabric calendars remain shared."""

    def __init__(self, layout, arrays):
        self.layout = layout
        self.arrays = arrays
        self.dirty_units = np.zeros(len(arrays[0]), dtype=np.bool_)
        self.dirty_banks = np.zeros(len(arrays[2]), dtype=np.bool_)

    def checkpoint(self):
        return (tuple(array.copy() for array in self.arrays),
                self.dirty_units.copy(), self.dirty_banks.copy())

    def restore(self, checkpoint):
        arrays, units, banks = checkpoint
        self.arrays = tuple(array.copy() for array in arrays)
        self.dirty_units, self.dirty_banks = units.copy(), banks.copy()

    def publish(self, cache, array_frontiers, buffer_frontiers, *, checkpoint=None, core=None):
        from .nand_core import _ReadCacheSlot

        arrays, units, banks = ((self.arrays, self.dirty_units, self.dirty_banks)
                               if checkpoint is None else checkpoint)
        array_ready, buffer_ready, tags, ready, readers, uses, _present = arrays
        bank_keys, tag_prefixes, array_names, pages_per_block = self.layout
        array_values, buffer_values = array_ready.tolist(), buffer_ready.tolist()
        for unit in np.flatnonzero(units).tolist():
            name = array_names[unit]
            if core is not None:
                core._touch_frontiers(name)
            array_frontiers[name], buffer_frontiers[name] = array_values[unit], buffer_values[unit]
        tag_values, ready_values = tags.tolist(), ready.tolist()
        reader_values, use_values = readers.tolist(), uses.tolist()
        for bank in np.flatnonzero(banks).tolist():
            key = bank_keys[bank]
            if core is not None and core._nand_transactions:
                core._touch_cache_bank(key)
            slots = cache.get(key)
            if slots is None:
                slots = [_ReadCacheSlot(), _ReadCacheSlot()]
                cache[key] = slots
            for index, slot in enumerate(slots):
                page = tag_values[bank][index]
                slot.page_key = None if page < 0 else (*tag_prefixes[bank], *divmod(page, pages_per_block))
                slot.ready_ns = ready_values[bank][index]
                slot.reader_completion_ns, slot.last_use = reader_values[bank][index], use_values[bank][index]
